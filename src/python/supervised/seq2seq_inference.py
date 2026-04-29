import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import numpy as np
from seq2seq import Seq2Seq, Encoder, Decoder
import math

def inference(model, iterator, force_predictions=False):
    device=model.device
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating..."):
            input_embeds = batch["input_embeds"].to(device)
            batch_predictions = model(input_embeds)
            if force_predictions:
                pred_y = torch.argmax(batch_predictions[:, :, :-1], dim=-1).detach().cpu().to(torch.int64)
            else:
                pred_y = torch.argmax(batch_predictions, dim=-1).detach().cpu().to(torch.int64)
            predictions.append(pred_y)
    # concat batches along batch dimension -> [N, T]
    preds = torch.cat(predictions, dim=0)  # shape consistent even if last batch smaller
    return preds.numpy()

# Dataset class
# there will be multiple input files to be memory-mapped with numpy
# and the labels are a part of the regular input files, not a separate window
class LabeledDatasetInference(Dataset):
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
        windows = [] # file idx, window step idx

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

        if ip.shape[-1] < self.num_parents+1:
            n_missing_cols = self.num_parents+1 - ip.shape[-1]
            add_empty_parents = np.concat([ip[:, :-1], np.zeros((ip.shape[0], n_missing_cols)), ip[:, -1].reshape(-1, 1)], axis=1)
            matrix = add_empty_parents[:, 0:self.num_parents+1]
        else:
            matrix = ip[:, 0:self.num_parents+1]
        labels = torch.tensor(matrix[:, -1], dtype=torch.int64)
        labels[labels == -1] = 24

        return {
            "input_embeds": torch.tensor(matrix[:, :-1], dtype=torch.float),
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
    parser.add_argument("--force-preds", action="store_true", help="forces predictions / disables unknown option")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initializing model
    ckpt = torch.load(args.model_path, map_location=device)
    EMB_DIM = args.embedding_dim
    HID_DIM = args.hidden_dim

    enc = Encoder(args.num_parents, EMB_DIM, HID_DIM)
    dec = Decoder(args.num_parents+1, EMB_DIM, HID_DIM)
    model = Seq2Seq(enc, dec, device)
    model.load_state_dict(ckpt)
    model.to(device)

    os.makedirs(args.save_dir, exist_ok=True)

    filenames = os.listdir(args.data_path)
    for file in filenames:
        dataset = LabeledDatasetInference(args.data_path, [file], args.max_seq_length, args.num_parents, args.step_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        predictions = inference(model, dataloader, args.force_preds).flatten()
        pred_file = file.split("_matrix")[0]
        ps4g_len = len(np.load(f"{args.data_path}/{file}", allow_pickle=True))
        assert(len(predictions) >= ps4g_len)
        np.save(os.path.join(args.save_dir, pred_file), predictions[:ps4g_len])


if __name__ == "__main__":
    main()