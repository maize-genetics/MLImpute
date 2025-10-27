import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import ModernBertDecoderForCausalLM, ModernBertDecoderConfig, get_wsd_schedule
import numpy as np
import wandb
from tqdm import tqdm

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
        embedded2 = self.dropout(self.bert(inputs_embeds=embedded, output_hidden_states=True).hidden_states[-1])  # (batch_size, seq_len, embedding_dim)
        predictions = self.fc(self.dropout(embedded2)) # (batch_size, seq_len, parent_dim)
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

    batch_size = 8
    
    optimizer.zero_grad()

    for idx, batch in enumerate(tqdm(iterator, desc="Training...")):
        if (idx/batch_size) % steps_per_round == 0: # start new round of WSD
            # TODO: make it so that the last round is as long as it takes to get through the iterator
            # so that we guarantee we end in a valley
            lr_scheduler = get_wsd_schedule(optimizer, num_warmup_steps, num_decay_steps,
                                            num_stable_steps=num_stable_steps)

        input_embeds = batch["input_embeds"].to(device)
        labels = batch["labels"].to(device)
        #optimizer.zero_grad()

        predictions = model(input_embeds)
        loss = criterion(predictions.permute(0, 2, 1), labels) / batch_size
        acc = path_acc(predictions.detach().cpu(), labels.detach().cpu())
        loss.backward()

        if (idx + 1) % batch_size == 0 or idx + 1 == len(iterator):

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()
        epoch_acc += acc

        if lr_scheduler.state_dict()["_step_count"] == steps_per_round - 1 and (idx+1) % batch_size == 0:
            wandb.log({"Valley Loss": epoch_loss / ((idx+1) / batch_size), "Valley Accuracy": epoch_acc / (idx+1),
                       "Step": (idx // batch_size)})
        elif (idx / batch_size) % 100 == 0 and lr_scheduler.state_dict()["_step_count"] > num_warmup_steps:
            wandb.log({"Hillside Loss": epoch_loss / ((idx + 1) / batch_size), "Hillside Accuracy": epoch_acc / (idx + 1),
                       "Step": (idx // batch_size)})


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

n_epochs = 9
num_parents = 25
max_sequence_length = 512
input_numpy = "/workdir/ahb232/MLImpute/src/training_data/train/CML442_matrix.npy"
input_labels = "/workdir/ahb232/MLImpute/src/CML442_test_labels.npy"
device="cuda"

configuration = ModernBertDecoderConfig(num_hidden_layers=2, max_position_embeddings=max_sequence_length)
model = BertTagger(ModernBertDecoderForCausalLM(configuration), num_parents, 0.1)


dataset = LabeledDataset(input_numpy, input_labels, max_sequence_length, num_parents)
dataset_chunks = torch.utils.data.random_split(dataset, [0.1] * 10)

optimizer = optim.AdamW(model.parameters())

criterion = nn.CrossEntropyLoss()

model.to(device)
criterion.to(device)

wandb.init(project="Test BERT tagger pipeline", name="test-causal-lm-4", config={
        "epochs": n_epochs,
        "batch_size": 8,
        "learning_rate": "WSD"
    })

best_loss = float('inf')

for epoch in range(n_epochs):
    print("Epoch " + str(epoch))
    dataloader = DataLoader(dataset_chunks[epoch], batch_size=1, shuffle=True)
    test_dataloader = DataLoader(dataset_chunks[-1], batch_size=1, shuffle=False)
    train_loss, train_acc = train(model, dataloader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_dataloader, criterion)

    wandb.log({"Epoch Loss": train_loss, "Epoch Accuracy": train_acc,
               "Test Loss": test_loss, "Test Accuracy": test_acc,
               "Epoch": epoch})

    if test_loss < best_loss:
        torch.save(model.state_dict(), "best_model.pt")

wandb.finish()

