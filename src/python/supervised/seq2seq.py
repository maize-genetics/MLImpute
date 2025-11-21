import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm
import argparse
import os
from torch.utils.data import DataLoader
from train_supervised import LabeledDataset, path_acc, evaluate

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
        predictions = torch.argmax(logits, dim=-1)
        diff = (predictions[:-1] == predictions[1:])
        smoothness_penalty = torch.sum(diff.float())
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
    criterion = nn.CrossEntropyLoss()

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