import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm
import argparse
import os
from torch.utils.data import DataLoader, Dataset
from train_supervised import path_acc

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
        windows = [] # file idx, window step idx

        for idx in range(len(self.input_file_names)):
            filelen = np.load(f"{self.file_dir}/{self.input_file_names[idx]}").shape[0]
            num_windows = (filelen - self.window_size) // self.step_size
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
        ip = np.load(f"{self.file_dir}/{self.input_file_names[file_idx]}", allow_pickle=True, mmap_mode='r')[pos_start:pos_end]
        if ip.shape[1] != 26: # copy haploid labels
            matrix = np.concatenate([ip, ip[:, self.num_parents:self.num_parents+1]], axis=1)
        else: # diploid labels
            matrix = ip[:, 0:self.num_parents+2]
        labels = torch.tensor(matrix[:, self.num_parents:self.num_parents+2], dtype=torch.int64)
        labels[labels == -1] = 24

        return {
            "input_embeds": torch.tensor(matrix[:, :-2], dtype=torch.float),
            "labels": labels
        }

class Encoder(nn.Module):
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

class Decoder(nn.Module):
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

class Seq2Seq(nn.Module):
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

class DiploidCrossEntropyLoss(nn.Module):
    """
    Cross-entropy over ploidy:
      logits:  (B, T, P, V)
      targets:(B, T, P)
    Equivalent to running CE over P*T tokens per batch.
    """
    def __init__(self, ploidy=2):
        super().__init__()
        self.ploidy = ploidy
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # logits: (B, T, P, V)
        # targets: (B, T, P)
        B, T, P, V = logits.shape
        assert P == self.ploidy, "Ploidy mismatch between logits and loss."

        # (B, T, P, V) -> (B, P, V, T) -> (B*P, V, T)
        logits = logits.permute(0, 2, 3, 1).reshape(B * P, V, T)

        # (B, T, P) -> (B, P, T) -> (B*P, T)
        targets = targets.permute(0, 2, 1).reshape(B * P, T)

        return self.ce(logits, targets)

class SmoothPredictDiploid(nn.Module):
    """
    Optional smoothing loss for diploid predictions.
    By default this *penalizes* positions where consecutive time steps
    have the same predictions (keeps original sign convention).
    """
    def __init__(self, lambda_smooth=0.2, ploidy=2):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.ce = DiploidCrossEntropyLoss(ploidy=ploidy)

    def forward(self, logits, targets):
        # logits: (B, T, P, V)
        # targets: (B, T, P)
        ce_loss = self.ce(logits, targets)

        # split into hap1 and hap2
        preds_hap1 = logits[:, :, 0].argmax(dim=-1)  # (B, T, 1)
        preds_hap2 = logits[:, :, 1].argmax(dim=-1)  # (B, T, 1)

        if preds_hap1.size(1) <= 1 or preds_hap2.size(1) <= 1:
            return ce_loss

        # compare adjacent time steps for each haplotype
        diff_hap1 = (preds_hap1[:-1] != preds_hap1[1:])
        diff_hap2 = (preds_hap2[:-1] != preds_hap2[1:])

        smoothness_penalty = diff_hap1.float().sum() + diff_hap2.float().sum()

        return ce_loss + self.lambda_smooth * smoothness_penalty

# training loop
def train(model, iterator, optimizer, criterion, steps_to_print):
    epoch_loss = 0
    epoch_acc = 0

    device = model.device
    model.train()

    for idx, batch in enumerate(tqdm(iterator, desc="Training...")):
        input_embeds = batch["input_embeds"].to(device)
        labels = batch["labels"].to(device)   # (batch, seq_len, ploidy)

        optimizer.zero_grad()

        # forward
        predictions = model(input_embeds)     # (batch, seq_len, ploidy, vocab)

        loss = criterion(predictions, labels)

        # You’ll likely want to update path_acc to handle ploidy, but for now you
        # can compute accuracy per haplotype similarly by flattening.
        # For a quick placeholder, here’s a naive genotype-accuracy:
        with torch.no_grad():
            # predictions: (batch, seq_len, ploidy, vocab)
            pred_labels = predictions.argmax(dim=-1)  # (batch, seq_len, ploidy)
            # exact-match both haplotypes at each position
            acc = path_acc(predictions.detach().cpu(), labels.detach().cpu())

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

        if idx % steps_to_print == 0:
            wandb.log({"Loss": epoch_loss / (idx+1), "Accuracy": epoch_acc / (idx+1), "Step": idx})

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

#evaluation loop
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    device=model.device
    model.eval()
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating..."):
            input_embeds = batch["input_embeds"].to(device)
            labels = batch["labels"].to(device)

            predictions = model(input_embeds)
            loss = criterion(predictions, labels)
            acc = path_acc(predictions.detach().cpu(), labels.detach().cpu())

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# arguments for running the script
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-parents", "--np", type=int, default=24, help="number of parents")
    parser.add_argument("--max-seq-length", "--sl", type=int, default=512, help="maximum input sequence length")
    parser.add_argument("--training-data-path", type=str, required=True, help="path to the input training data")
    parser.add_argument("--validation-data-path", type=str, required=True, help="path to the input validation data")
    parser.add_argument("--num-epochs", "-e", type=int, default=9, help="number of training epochs")
    parser.add_argument("--step-size", "-s", type=int, default=128, help="distance between the start points of each training window")
    parser.add_argument("--project-name", "--pn", type=str, default="test", help="wandb project name")
    parser.add_argument("--run-name", "--rn", type=str, default="run-1", help="wand run name")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="batch size")
    parser.add_argument("--save-model-path", type=str, default="best_model.pt", help="path to save the best performing model")
    parser.add_argument("--steps-to-print", "--sp", type=int, default=100, help="steps between reporting to wandb")
    parser.add_argument("--embedding-dim", type=int, default=12, help="embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=24, help="hidden dimension")
    parser.add_argument("--ls", type=float, default=0.0, help="smoothing hyperparameter")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Initializing model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EMB_DIM = args.embedding_dim
    HID_DIM = args.hidden_dim
    ploidy = 2

    enc = Encoder(args.num_parents, EMB_DIM, HID_DIM)
    dec = Decoder(args.num_parents + 1, EMB_DIM, HID_DIM, ploidy=ploidy)
    model = Seq2Seq(enc, dec, device, ploidy=ploidy)

    # Initializing training dataset
    training_filenames = os.listdir(args.training_data_path)
    training_dataset = LabeledDatasetDiploid(args.training_data_path, training_filenames, args.max_seq_length, args.num_parents, args.step_size)

    # Initializing validation dataset
    validation_filenames = os.listdir(args.validation_data_path)
    validation_dataset = LabeledDatasetDiploid(args.validation_data_path, validation_filenames, args.max_seq_length, args.num_parents, args.step_size)

    # Setting up optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    #criterion = nn.CrossEntropyLoss()
    criterion = SmoothPredictDiploid(lambda_smooth=args.ls)

    model.to(device)
    criterion.to(device)

    # start up wandb run
    wandb.init(project=args.project_name, entity="maize-genetics", name=args.run_name, config={
            "epochs": args.num_epochs,
            "batch_size": args.batch_size
        })

    best_loss = float('inf')

    # loop through epochs for training
    for epoch in range(args.num_epochs):
        dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
        train_loss, train_acc = train(model, dataloader, optimizer, criterion, args.steps_to_print)
        test_loss, test_acc = evaluate(model, test_dataloader, criterion)

        wandb.log({"Epoch Loss": train_loss, "Epoch Accuracy": train_acc,
                   "Test Loss": test_loss, "Test Accuracy": test_acc,
                   "Epoch": epoch})

        if test_loss < best_loss:
            torch.save(model.state_dict(), args.save_model_path)

    wandb.finish()

if __name__ == '__main__':
    main()