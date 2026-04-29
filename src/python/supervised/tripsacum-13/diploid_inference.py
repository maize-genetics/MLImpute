import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import math

class EncoderDiploid(nn.Module):
    def __init__(self, in_features, emb_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(in_features, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)

    def forward(self, src_feats):
        """
        src_feats: (seq_len, batch_size, in_features)  # in_features = 24
        """
        emb = self.proj(src_feats)          # (seq_len, batch, emb_dim)
        outputs, hidden = self.rnn(emb)     # hidden: (1, batch, hidden_dim)
        return hidden

class DecoderDiploid(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, ploidy=2):
        super().__init__()
        self.output_dim = output_dim      # vocab_size = num_parents + 1
        self.ploidy = ploidy              # 2 for diploid

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)
        # predict ploidy * vocab logits from hidden
        self.fc = nn.Linear(hidden_dim, output_dim * ploidy)

    def forward(self, input, hidden):
        """
        input: (batch, ploidy) integer tokens for each haplotype
        hidden: (1, batch, hidden_dim)
        returns:
            prediction: (batch, ploidy, output_dim)
            hidden:     (1, batch, hidden_dim)
        """
        # input: (batch, ploidy) -> (batch, ploidy, emb_dim)
        embedded = self.embedding(input)

        # combine haplotypes into one context vector
        # (batch, ploidy, emb_dim) -> (batch, emb_dim)
        embedded = embedded.sum(dim=1)

        # GRU expects (seq_len=1, batch, emb_dim)
        embedded = embedded.unsqueeze(0)

        output, hidden = self.rnn(embedded, hidden)  # output: (1, batch, hidden_dim)
        output = output.squeeze(0)                   # (batch, hidden_dim)

        # (batch, hidden_dim) -> (batch, ploidy * vocab)
        prediction = self.fc(output)
        prediction = prediction.view(-1, self.ploidy, self.output_dim)  # (batch, ploidy, vocab)

        return prediction, hidden

class Seq2SeqDiploid(nn.Module):
    def __init__(self, encoder, decoder, device, ploidy=2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.ploidy = ploidy

    def forward(self, src_feats, trg=None, teacher_forcing_ratio=0.5):
        """
        src_feats: (batch, seq_len, in_features)
        trg:       (batch, seq_len, ploidy) or None
        returns:
            outputs: (batch, seq_len, ploidy, vocab_size)
        """
        batch_size, seq_len, _ = src_feats.shape
        vocab_size = self.decoder.output_dim

        # rearrange to (seq_len, batch, features) for GRU encoder
        src_feats = src_feats.permute(1, 0, 2)   # (seq_len, batch, in_features)

        if trg is not None:
            # (batch, seq_len, ploidy) -> (seq_len, batch, ploidy)
            trg = trg.permute(1, 0, 2)

        # Encoder
        hidden = self.encoder(src_feats)        # (1, batch, hidden_dim)

        # Initial input token pair to decoder, e.g. all zeros
        input = torch.zeros(batch_size, self.ploidy, dtype=torch.long, device=self.device)

        outputs = []

        for t in range(seq_len):
            # output: (batch, ploidy, vocab)
            output, hidden = self.decoder(input, hidden)
            # store with time dimension
            outputs.append(output.unsqueeze(1))   # (batch, 1, ploidy, vocab)

            use_teacher_forcing = (trg is not None) and (torch.rand(1).item() < teacher_forcing_ratio)
            if use_teacher_forcing:
                # trg[t]: (batch, ploidy)
                input = trg[t]
            else:
                # argmax over vocab -> (batch, ploidy)
                input = output.argmax(dim=-1)

        # concat over time: list[(batch, 1, ploidy, vocab)] -> (batch, seq_len, ploidy, vocab)
        outputs = torch.cat(outputs, dim=1)

        return outputs

def inference(model, iterator, force_predictions=False):
    device=model.device
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating..."):
            input_embeds = batch["input_embeds"].to(device)
            batch_predictions = model(input_embeds)
            if force_predictions:
                pred_y = torch.argmax(batch_predictions[:, :, :, :13], dim=-1).detach().cpu().to(torch.int64)
            else:
                pred_y = torch.argmax(batch_predictions, dim=-1).detach().cpu().to(torch.int64)
            predictions.append(pred_y)
    # concat batches along batch dimension -> [N, T]
    preds = torch.cat(predictions, dim=0)  # shape consistent even if last batch smaller
    return preds.numpy()

# Dataset class
# there will be multiple input files to be memory-mapped with numpy
# and the labels are a part of the regular input files, not a separate window
class LabeledDatasetDiploid(Dataset):
    def __init__(self, file_dir, input_file_names, window_size=512, num_parents=24, step_size=128):
        self.file_dir = file_dir
        self.input_file_names = input_file_names
        self.window_size = window_size
        self.num_parents = num_parents
        self.step_size = step_size
        self.windows = self.__generate_windows__()

        self.n_windows = len(self.windows)

    # uses a list to store all valid windows
    # this could be manipulated to skip windows with unlabeled bins
    def __generate_windows__(self):
        # windows is a list of tuples, where each tuple represents a training data point
        # the tuples are formatted as (file index, window step index)
        # multiply window step index by step_size to get the index of the first position in the window
        windows = []  # file idx, window step idx

        for idx in range(len(self.input_file_names)):
            filelen = np.load(f"{self.file_dir}/{self.input_file_names[idx]}").shape[0]
            num_windows = math.ceil(filelen / self.step_size)
            windows.extend([(idx, idy) for idy in range(num_windows)])

        return windows

    def __len__(self):
        return self.n_windows

    # only required pieces of data are the input embeddings and the correct labels
    def __getitem__(self, idx):
        # retrieve window index from list
        file_idx, pos_idx = self.windows[idx]

        # convert to position start and end
        pos_start = pos_idx * self.step_size
        pos_end = pos_start + self.window_size

        # grab segment from mmaped numpy
        arr = np.load(
            f"{self.file_dir}/{self.input_file_names[file_idx]}",
            allow_pickle=True,
            mmap_mode="r",
        )

        ip = arr[pos_start:min(pos_end, arr.shape[0])]
        if ip.shape[0] < self.window_size:
            pad_len = self.window_size - ip.shape[0]
            pad = np.zeros((pad_len, ip.shape[1]), dtype=ip.dtype)
            pad[:, -1] = -1
            ip = np.concatenate([ip, pad], axis=0)

        add_empty_parents = np.concatenate(
            [
                ip[:, :-1],  # original parent columns
                np.zeros((ip.shape[0], 11)),  # empty diploid parents
                -np.ones((ip.shape[0], 1)),  # extra label = -1
                ip[:, -1].reshape(-1, 1)
            ],
            axis=1,
        )
        matrix = add_empty_parents[:, 0:self.num_parents+2]
        labels = torch.tensor(matrix[:, self.num_parents:self.num_parents+2], dtype=torch.int64)
        labels[labels == -1] = 24

        return {
            "input_embeds": torch.tensor(matrix[:, :-2], dtype=torch.float),
            "labels": labels
        }

# arguments for running the script
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, required=True, help="path to the input data")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="batch size")
    parser.add_argument("--model-path", type=str, default="best_model.pt", help="path to model")
    parser.add_argument("--save-dir", type=str, required=True, help="path to save imputed paths")
    parser.add_argument("--num-parents", "--np", type=int, default=24, help="number of parents")
    parser.add_argument("--max-seq-length", "--sl", type=int, default=512, help="maximum input sequence length")
    parser.add_argument("--step-size", "-s", type=int, default=128, help="distance between the start points of each training window")
    parser.add_argument("--embedding-dim", type=int, default=12, help="embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=24, help="hidden dimension")
    parser.add_argument("--force-preds", type=bool, default=False, help="forces predictions / disables unknown option")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initializing model
    ckpt = torch.load(args.model_path, map_location=device)
    EMB_DIM = args.embedding_dim
    HID_DIM = args.hidden_dim

    enc = EncoderDiploid(args.num_parents, EMB_DIM, HID_DIM)
    dec = DecoderDiploid(args.num_parents+1, EMB_DIM, HID_DIM, ploidy=2)
    model = Seq2SeqDiploid(enc, dec, device, ploidy=2)
    model.load_state_dict(ckpt)
    model.to(device)

    os.makedirs(args.save_dir, exist_ok=True)

    filenames = os.listdir(args.data_path)
    for file in filenames:
        dataset = LabeledDatasetDiploid(args.data_path, [file], args.max_seq_length, args.num_parents, args.step_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        predictions = inference(model, dataloader, args.force_preds)
        predictions = predictions.reshape(-1, predictions.shape[-1])
        pred_file = file.split("_matrix")[0]
        ps4g_len = len(np.load(f"{args.data_path}/{file}", allow_pickle=True))
        assert(predictions.shape[0] >= ps4g_len)
        np.save(os.path.join(args.save_dir, pred_file), predictions[:ps4g_len, :])


if __name__ == "__main__":
    main()