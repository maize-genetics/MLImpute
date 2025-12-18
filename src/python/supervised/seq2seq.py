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

# Dataset class
# there will be multiple input files to be memory-mapped with numpy
# and the labels are a part of the regular input files, not a separate window
class LabeledDataset(Dataset):
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

        matrix = ip[:, 0:self.num_parents+1]
        labels = torch.tensor(matrix[:, -1], dtype=torch.int64)
        labels[labels == -1] = 24

        return {
            "input_embeds": torch.tensor(matrix[:, :-1], dtype=torch.float),
            "labels": labels
        }

# Calculates the accuracy of the path (for a more practical measurement of performance than loss)
def path_acc(preds, labels):
    pred_y = np.argmax(preds, -1)
    num_correct = np.count_nonzero(pred_y == labels)
    return num_correct / torch.numel(labels)

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
            loss = criterion(predictions.permute(0, 2, 1), labels)
            acc = path_acc(predictions.detach().cpu(), labels.detach().cpu())

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

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
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)  # output_dim = vocab_size (=25)

    def forward(self, input, hidden):
        """
        input: (batch,) integer tokens
        hidden: (1, batch, hidden_dim)
        """
        input = input.unsqueeze(0)              # (1, batch)
        embedded = self.embedding(input)        # (1, batch, emb_dim)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output.squeeze(0)) # (batch, output_dim)
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src_feats, trg=None, teacher_forcing_ratio=0.5):
        """
        src_feats: (batch, seq_len, in_features=24)
        trg:       (batch, seq_len)  or None
        returns:
            outputs: (seq_len, batch, vocab_size)
        """
        batch_size, seq_len, _ = src_feats.shape
        vocab_size = self.decoder.fc.out_features

        # rearrange to (seq_len, batch, features) for GRU
        src_feats = src_feats.permute(1, 0, 2)   # (seq_len, batch, 24)

        if trg is not None:
            trg = trg.permute(1, 0)             # (seq_len, batch)

        # Encoder
        hidden = self.encoder(src_feats)        # (1, batch, hidden_dim)

        # Initial input token to decoder (start token, here just 0)
        input = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        outputs = []

        for t in range(seq_len):
            output, hidden = self.decoder(input, hidden)  # (batch, vocab)
            outputs.append(output.unsqueeze(0))           # (1, batch, vocab)

            if trg is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input = trg[t]                            # true token at time t
            else:
                input = output.argmax(dim=1)              # predicted token

        outputs = torch.cat(outputs, dim=0)               # (seq_len, batch, vocab)
        outputs = outputs.permute(1, 0, 2)
        return outputs

class SmoothPredict(nn.Module):
    def __init__(self, lambda_smooth=0.2):
        super(SmoothPredict, self).__init__()
        self.lambda_smooth = lambda_smooth
        self.cross_entropy = nn.CrossEntropyLoss()
    def __call__(self, logits, targets):
        # logits [batch_size, num_parents, sequence_length]
        # targets [batch_size, sequence_length]
        predictions = torch.argmax(logits, dim=1)
        # predictions [batch_size, sequence_length]
        diff = (predictions[:, :-1] != predictions[:, 1:])
        # diff [batch_size, sequence_length-1]
        smoothness_penalty = torch.mean(torch.sum(diff.float(), dim=1))
        ce_loss = self.cross_entropy(logits, targets)
        return ce_loss + self.lambda_smooth * smoothness_penalty


# training loop
def train(model, iterator, optimizer, criterion, steps_to_print):
    epoch_loss = 0
    epoch_acc = 0

    device = model.device

    model.train()

    # training loop itself
    for idx, batch in enumerate(tqdm(iterator, desc="Training...")):
        # load batch info
        input_embeds = batch["input_embeds"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()

        # forward and backward passes through the model
        predictions = model(input_embeds)
        loss = criterion(predictions.permute(0, 2, 1), labels)
        acc = path_acc(predictions.detach().cpu(), labels.detach().cpu())
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

        # update wandb regularly
        if idx % steps_to_print == 0:
            wandb.log({"Loss": epoch_loss / (idx+1), "Accuracy": epoch_acc / (idx+1), "Step": idx})

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

    enc = Encoder(args.num_parents, EMB_DIM, HID_DIM)
    dec = Decoder(args.num_parents+1, EMB_DIM, HID_DIM)
    model = Seq2Seq(enc, dec, device)

    # Initializing training dataset
    training_filenames = os.listdir(args.training_data_path)
    training_dataset = LabeledDataset(args.training_data_path, training_filenames, args.max_seq_length, args.num_parents, args.step_size)

    # Initializing validation dataset
    validation_filenames = os.listdir(args.validation_data_path)
    validation_dataset = LabeledDataset(args.validation_data_path, validation_filenames, args.max_seq_length, args.num_parents, args.step_size)

    # Setting up optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    #criterion = nn.CrossEntropyLoss()
    criterion = SmoothPredict(lambda_smooth=args.ls)

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