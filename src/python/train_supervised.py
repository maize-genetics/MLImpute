import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import ModernBertModel, ModernBertConfig, get_wsd_schedule
import numpy as np
import wandb
from tqdm import tqdm
import argparse

class BertTagger(nn.Module):
    def __init__(self, bert, parent_dim, dropout):
        super().__init__()

        self.bert = bert
        embedding_dim = bert.config.to_dict()["hidden_size"]
        self.embedding = nn.Linear(parent_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_parents)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        # input_ids: npy multi-hot embedding: (batch_size, seq_len, parent_dim)
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(self.bert(inputs_embeds=embedded)[0])  # (batch_size, seq_len, embedding_dim)
        predictions = self.fc(self.dropout(embedded)) # (batch_size, seq_len, parent_dim)
        return predictions


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

    def __getitem__(self, idx):
        pos_start = idx * self.step_size
        pos_end = pos_start + self.window_size

        return {
            "input_embeds": torch.tensor(self.matrix[pos_start:pos_end], dtype=torch.float),
            "labels": torch.tensor(self.labels[pos_start:pos_end], dtype=torch.int64)
        }

def path_acc(preds, labels):
    pred_y = np.argmax(preds, -1)

    num_correct = np.count_nonzero(pred_y == labels)

    return num_correct / torch.numel(labels)

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    num_warmup_steps = 20
    num_decay_steps = 20
    num_stable_steps = 200

    steps_until_decay = num_warmup_steps + num_stable_steps
    steps_per_round = steps_until_decay + num_decay_steps

    model.train()
    lr_scheduler = None

    for idx, batch in enumerate(tqdm(iterator, desc="Training...")):
        if idx % steps_per_round == 0: # start new round of WSD
            # so that we guarantee we end in a valley
            if len(iterator) - idx <= 2* steps_per_round:
                lr_scheduler = get_wsd_schedule(optimizer, num_warmup_steps, num_decay_steps,
                                                num_training_steps=len(iterator-idx))
            else:
                lr_scheduler = get_wsd_schedule(optimizer, num_warmup_steps, num_decay_steps,
                                            num_stable_steps=num_stable_steps)

        input_embeds = batch["input_embeds"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()

        predictions = model(input_embeds)
        loss = criterion(predictions.permute(0, 2, 1), labels)
        acc = path_acc(predictions.detach().cpu(), labels.detach().cpu())
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += acc

        if lr_scheduler.state_dict()["_step_count"] == steps_per_round - 1:
            wandb.log({"Valley Loss": epoch_loss / (idx+1), "Valley Accuracy": epoch_acc / (idx+1),
                       "Step": idx})
        elif idx % 100 == 0 and lr_scheduler.state_dict()["_step_count"] > num_warmup_steps:
            wandb.log({"Hillside Loss": epoch_loss / (idx + 1), "Hillside Accuracy": epoch_acc / (idx + 1),
                       "Step": idx})


    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-parents", "--np", type=int, default=25, help="number of parents")
    parser.add_argument("--max-seq-length", "--sl", type=int, default=512, help="maximum input sequence length")
    parser.add_argument("--num-epochs", "-e", type=int, default=9, help="number of training epochs")
    parser.add_argument("--num-hidden-layers", "--nh", type=int, default=2, help="number of hidden layers in BERT")
    parser.add_argument("--step-size", "-s", type=int, default=128, help="distance between the start points of each training window")
    parser.add_argument("--project-name", "--pn", type=str, default="test", help="wandb project name")
    parser.add_argument("--run-name", "--rn", type=str, default="run-1", help="wand run name")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="batch size")
    parser.add_argument("--save-model-path", "-s", type=str, default="best_model.pt", help="path to save the best performing model")
    parser.add_argument("--steps-to-print", "--sp", type=int, default=100, help="steps between reporting to wandb")
    parser.add_argument("--warmup-steps", "--warm", type=int, default=20, help="number of warmup steps")
    parser.add_argument("--stable-steps", "--stable", type=int, default=200, help="number of stable steps")
    parser.add_argument("--decay-steps", "--decay", type=int, default=20, help="number of decay steps")

    args = parser.parse_args()
    return args





n_epochs = 9
num_parents = 25
max_sequence_length = 512
input_numpy = "/workdir/ahb232/MLImpute/src/training_data/train/CML442_matrix.npy"
input_labels = "/workdir/ahb232/MLImpute/src/CML442_test_labels.npy"
device="cuda"

configuration = ModernBertConfig(num_hidden_layers=2, max_position_embeddings=max_sequence_length)
model = BertTagger(ModernBertModel(configuration), num_parents, 0.1)


dataset = LabeledDataset(input_numpy, input_labels, max_sequence_length, num_parents)
dataset_chunks = torch.utils.data.random_split(dataset, [0.1] * 10)

optimizer = optim.AdamW(model.parameters())

criterion = nn.CrossEntropyLoss()

model.to(device)
criterion.to(device)

wandb.init(project="Test BERT tagger pipeline", name="test-3", config={
        "epochs": n_epochs,
        "batch_size": 8,
        "learning_rate": "WSD"
    })

best_loss = float('inf')

for epoch in range(n_epochs):
    dataloader = DataLoader(dataset_chunks[epoch], batch_size=8, shuffle=True)
    test_dataloader = DataLoader(dataset_chunks[-1], batch_size=8, shuffle=False)
    train_loss, train_acc = train(model, dataloader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_dataloader, criterion)

    wandb.log({"Epoch Loss": train_loss, "Epoch Accuracy": train_acc,
               "Test Loss": test_loss, "Test Accuracy": test_acc,
               "Epoch": epoch})

    if test_loss < best_loss:
        torch.save(model.state_dict(), "best_model.pt")

wandb.finish()
