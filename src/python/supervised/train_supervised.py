# classification model based off of Part-of-Speech tagging
# to be used as a baseline comparison against segmentation model
# Code adapted from this PoS Tagging tutorial: https://github.com/sejas/pytorch-pos-tagging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import ModernBertModel, ModernBertConfig, get_wsd_schedule
import numpy as np
import wandb
from tqdm import tqdm
import argparse


# Model architecture
# Effectively just a modernbert model with a simple classifier head and some dropout
class BertTagger(nn.Module):
    def __init__(self, bert, parent_dim, dropout):
        super().__init__()

        self.bert = bert
        embedding_dim = bert.config.to_dict()["hidden_size"]
        self.embedding = nn.Linear(parent_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, parent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        # input_ids: npy multi-hot embedding: (batch_size, seq_len, parent_dim)
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(self.bert(inputs_embeds=embedded)[0])  # (batch_size, seq_len, embedding_dim)
        predictions = self.fc(self.dropout(embedded)) # (batch_size, seq_len, parent_dim)
        return predictions

# Dataset class
# TODO: input file handling should be changed
# there will be multiple input files to be memory-mapped with numpy
# and the labels are a part of the regular input files, not a separate window
class LabeledDataset(Dataset):
    def __init__(self, input_file, label_file, window_size=512, top_n=25, step_size=128):
        self.window_size = window_size
        self.top_n = top_n
        self.step_size = step_size
        self.matrix = np.load(input_file)[:, 0:top_n]
        self.labels = np.load(label_file)

        self.n_windows = (self.matrix.shape[0] - window_size) // step_size

    def __len__(self):
        return self.n_windows

    # only required pieces of data are the input embeddings and the correct labels
    def __getitem__(self, idx):
        pos_start = idx * self.step_size
        pos_end = pos_start + self.window_size

        return {
            "input_embeds": torch.tensor(self.matrix[pos_start:pos_end], dtype=torch.float),
            "labels": torch.tensor(self.labels[pos_start:pos_end], dtype=torch.int64)
        }


# Calculates the accuracy of the path (for a more practical measurement of performance than loss)
def path_acc(preds, labels):
    pred_y = np.argmax(preds, -1)
    num_correct = np.count_nonzero(pred_y == labels)
    return num_correct / torch.numel(labels)


# training loop
def train(model, iterator, optimizer, criterion, num_warmup_steps,num_stable_steps, num_decay_steps, steps_to_print):
    epoch_loss = 0
    epoch_acc = 0

    # this is needed for WSD scheduling
    steps_until_decay = num_warmup_steps + num_stable_steps
    steps_per_round = steps_until_decay + num_decay_steps

    device = model.device

    model.train()
    lr_scheduler = None

    # training loop itself
    for idx, batch in enumerate(tqdm(iterator, desc="Training...")):

        # if we have gone through a round of WSD, we need to re-initialize the scheduler
        # because it doesn't appear to have the option to cycle itself
        if idx % steps_per_round == 0: # start new round of WSD

            # if we're on the last round, we increase the number of training steps total
            # so that we guarantee we end in a valley at the final batch
            if len(iterator) - idx <= 2* steps_per_round:
                lr_scheduler = get_wsd_schedule(optimizer, num_warmup_steps, num_decay_steps,
                                                num_training_steps=len(iterator-idx))
            else:
                lr_scheduler = get_wsd_schedule(optimizer, num_warmup_steps, num_decay_steps,
                                            num_stable_steps=num_stable_steps)

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
        lr_scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += acc

        # update wandb regularly
        if lr_scheduler.state_dict()["_step_count"] == steps_per_round - 1:
            wandb.log({"Valley Loss": epoch_loss / (idx+1), "Valley Accuracy": epoch_acc / (idx+1),
                       "Step": idx})
        elif idx % steps_to_print == 0 and lr_scheduler.state_dict()["_step_count"] > num_warmup_steps:
            wandb.log({"Hillside Loss": epoch_loss / (idx + 1), "Hillside Accuracy": epoch_acc / (idx + 1),
                       "Step": idx})


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
            loss = criterion(predictions.permute(0, 2, 1), labels)
            acc = path_acc(predictions.detach().cpu(), labels.detach().cpu())

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# arguments for running the script
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-parents", "--np", type=int, default=24, help="number of parents")
    parser.add_argument("--max-seq-length", "--sl", type=int, default=512, help="maximum input sequence length")
    parser.add_argument("--data-file-name", type=str, required=True, help="path to the input training data")
    parser.add_argument("--label-file-name", type=str, required=True, help="path to the input labels")
    parser.add_argument("--num-epochs", "-e", type=int, default=9, help="number of training epochs")
    parser.add_argument("--num-hidden-layers", "--nh", type=int, default=12, help="number of hidden layers in BERT")
    parser.add_argument("--step-size", "-s", type=int, default=128, help="distance between the start points of each training window")
    parser.add_argument("--project-name", "--pn", type=str, default="test", help="wandb project name")
    parser.add_argument("--run-name", "--rn", type=str, default="run-1", help="wand run name")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="batch size")
    parser.add_argument("--save-model-path", "-s", type=str, default="best_model.pt", help="path to save the best performing model")
    parser.add_argument("--steps-to-print", "--sp", type=int, default=100, help="steps between reporting to wandb")
    parser.add_argument("--warmup-steps", "--warm", type=int, default=20, help="number of warmup steps")
    parser.add_argument("--stable-steps", "--stable", type=int, default=200, help="number of stable steps. Should probably be larger than default (200)")
    parser.add_argument("--decay-steps", "--decay", type=int, default=20, help="number of decay steps. Should be about 10% of runtime")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device="cuda"

    # Initializing model
    configuration = ModernBertConfig(num_hidden_layers=args.num_hidden_layers,
                                     max_position_embeddings=args.max_seq_length)
    model = BertTagger(ModernBertModel(configuration), args.num_parents, 0.1)

    # Initializing dataset
    dataset = LabeledDataset(args.data_file_name, args.label_file_name, args.max_seq_length,
                             args.num_parents, args.step_size)
    dataset_chunks = torch.utils.data.random_split(dataset, [1 / (args.num_epochs + 1)] * (args.num_epochs + 1))

    # Setting up optimizer and loss function
    optimizer = optim.AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)

    # start up wandb run
    wandb.init(project=args.project_name, name=args.run_name, config={
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": "WSD"
        })

    best_loss = float('inf')

    # loop through epochs for training
    for epoch in range(args.num_epochs):
        dataloader = DataLoader(dataset_chunks[epoch], batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset_chunks[-1], batch_size=args.batch_size, shuffle=False)
        train_loss, train_acc = train(model, dataloader, optimizer, criterion, args.warmup_steps,
                                      args.stable_steps, args.decay_steps, args.steps_to_print)
        test_loss, test_acc = evaluate(model, test_dataloader, criterion)

        wandb.log({"Epoch Loss": train_loss, "Epoch Accuracy": train_acc,
                   "Test Loss": test_loss, "Test Accuracy": test_acc,
                   "Epoch": epoch})

        if test_loss < best_loss:
            torch.save(model.state_dict(), args.save_model_path)

    wandb.finish()

if __name__ == '__main__':
    main()