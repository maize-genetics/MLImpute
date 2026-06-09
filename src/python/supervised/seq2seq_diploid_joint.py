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


# Calculates unordered diploid accuracy at each position.
def path_acc_diploid(preds, labels):
    p1, p2 = preds[:, :, 0], preds[:, :, 1]
    l1, l2 = labels[:, :, 0], labels[:, :, 1]

    correct = ((p1 == l1) & (p2 == l2)) | ((p1 == l2) & (p2 == l1))
    return correct.float().mean()


# Convert joint pair logits into diploid parent predictions.
# logits: (B, T, V, V)
# returns: (B, T, 2)
def decode_joint_predictions(logits):
    B, T, V, V2 = logits.shape
    assert V == V2, "Joint logits must be square over vocab x vocab."

    pair_idx = logits.reshape(B, T, V * V).argmax(dim=-1)  # (B, T)
    p1 = pair_idx // V
    p2 = pair_idx % V
    return torch.stack([p1, p2], dim=-1)  # (B, T, 2)


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
        ip = np.load(
            f"{self.file_dir}/{self.input_file_names[file_idx]}",
            allow_pickle=True,
            mmap_mode='r'
        )[pos_start:pos_end]

        if ip.shape[1] != self.num_parents+2:  # copy haploid labels
            matrix = np.concatenate([ip, ip[:, self.num_parents:self.num_parents+1]], axis=1)
        else:  # diploid labels
            matrix = ip[:, 0:self.num_parents+2]

        labels = torch.tensor(matrix[:, self.num_parents:self.num_parents+2], dtype=torch.int64)
        labels[labels == -1] = self.num_parents

        return {
            "input_embeds": torch.tensor(matrix[:, :-2], dtype=torch.float),
            "labels": labels
        }


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


class DecoderDiploidJoint(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, ploidy=2):
        super().__init__()
        self.output_dim = output_dim      # vocab_size = num_parents + 1
        self.ploidy = ploidy              # kept for input token shape, expected 2

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)
        # predict joint pair logits over vocab x vocab
        self.fc = nn.Linear(hidden_dim, output_dim * output_dim)

    def forward(self, input, hidden):
        """
        input: (batch, ploidy) integer tokens for each haplotype
        hidden: (1, batch, hidden_dim)
        returns:
            prediction: (batch, output_dim, output_dim)
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

        # (batch, hidden_dim) -> (batch, vocab * vocab)
        prediction = self.fc(output)
        prediction = prediction.view(-1, self.output_dim, self.output_dim)  # (batch, vocab, vocab)

        return prediction, hidden


class Seq2SeqDiploidJoint(nn.Module):
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
            outputs: (batch, seq_len, vocab_size, vocab_size)
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
            # output: (batch, vocab, vocab)
            output, hidden = self.decoder(input, hidden)
            # store with time dimension
            outputs.append(output.unsqueeze(1))   # (batch, 1, vocab, vocab)

            use_teacher_forcing = (trg is not None) and (torch.rand(1).item() < teacher_forcing_ratio)
            if use_teacher_forcing:
                # trg[t]: (batch, ploidy)
                input = trg[t]
            else:
                # decode best joint pair -> (batch, ploidy)
                pair_idx = output.reshape(batch_size, vocab_size * vocab_size).argmax(dim=-1)
                p1 = pair_idx // vocab_size
                p2 = pair_idx % vocab_size
                input = torch.stack([p1, p2], dim=-1)

        # concat over time: list[(batch, 1, vocab, vocab)] -> (batch, seq_len, vocab, vocab)
        outputs = torch.cat(outputs, dim=1)

        return outputs


class JointPairCrossEntropyLoss(nn.Module):
    """
    Swap-invariant CE for joint diploid pair prediction.

    logits:  (B, T, V, V)
    targets: (B, T, 2)

    Predicts a joint distribution over parent pairs:
      P(parent_i, parent_j)

    Since diploid parent order is not meaningful, this loss uses the better
    of (p1, p2) and (p2, p1) at each position.
    """
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction  # "mean", "sum", or "none"

    def forward(self, logits, targets):
        B, T, V, V2 = logits.shape
        assert V == V2, "Joint pair logits must have shape (B, T, V, V)."
        assert targets.shape[-1] == 2, "Targets must have ploidy dimension 2."

        logits_flat = logits.reshape(B * T, V * V)   # (B*T, V*V)

        p1 = targets[:, :, 0].reshape(B * T)
        p2 = targets[:, :, 1].reshape(B * T)

        idx_forward = p1 * V + p2
        idx_reverse = p2 * V + p1

        loss_fwd = F.cross_entropy(logits_flat, idx_forward, reduction="none")
        loss_rev = F.cross_entropy(logits_flat, idx_reverse, reduction="none")

        loss = torch.minimum(loss_fwd, loss_rev)  # (B*T,)

        if self.reduction == "none":
            return loss.reshape(B, T)
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


class SmoothPredictDiploidJoint(nn.Module):
    """
    Optional smoothing loss for joint diploid pair predictions.

    logits:  (B, T, V, V)
    targets: (B, T, 2)

    Note: because the pair is unordered, smoothing across the two decoded
    haplotypes can be less stable than in an ordered-haplotype model. This is
    included mainly to preserve parity with the old script. For your first test
    of joint prediction, setting --ls 0.0 is a good idea.
    """
    def __init__(self, lambda_smooth=0.2):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.ce = JointPairCrossEntropyLoss(reduction="mean")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        pred_labels = decode_joint_predictions(logits)  # (B, T, 2)
        preds_hap1 = pred_labels[:, :, 0]               # (B, T)
        preds_hap2 = pred_labels[:, :, 1]               # (B, T)

        if preds_hap1.size(1) <= 1 or preds_hap2.size(1) <= 1:
            return ce_loss

        diff_hap1 = (preds_hap1[:, :-1] != preds_hap1[:, 1:])
        diff_hap2 = (preds_hap2[:, :-1] != preds_hap2[:, 1:])

        smoothness_penalty = (
            torch.mean(torch.sum(diff_hap1.float(), dim=1)) +
            torch.mean(torch.sum(diff_hap2.float(), dim=1))
        )

        return ce_loss + self.lambda_smooth * smoothness_penalty


# training loop
def train(model, iterator, optimizer, criterion, steps_to_print):
    epoch_loss = 0
    epoch_acc = 0

    device = model.device
    model.train()

    for idx, batch in enumerate(tqdm(iterator, desc="Training...")):
        input_embeds = batch["input_embeds"].to(device)
        labels = batch["labels"].to(device)   # (batch, seq_len, 2)

        optimizer.zero_grad()

        # forward
        predictions = model(input_embeds)     # (batch, seq_len, vocab, vocab)

        loss = criterion(predictions, labels)

        with torch.no_grad():
            pred_labels = decode_joint_predictions(predictions)  # (batch, seq_len, 2)
            acc = path_acc_diploid(pred_labels, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if idx % steps_to_print == 0:
            wandb.log({
                "Loss": epoch_loss / (idx + 1),
                "Accuracy": epoch_acc / (idx + 1),
                "Step": idx
            })

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# evaluation loop
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    device = model.device
    model.eval()

    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating..."):
            input_embeds = batch["input_embeds"].to(device)
            labels = batch["labels"].to(device)

            predictions = model(input_embeds)   # (batch, seq_len, vocab, vocab)
            loss = criterion(predictions, labels)

            pred_labels = decode_joint_predictions(predictions)  # (batch, seq_len, 2)
            acc = path_acc_diploid(pred_labels, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

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
    parser.add_argument("--run-name", "--rn", type=str, default="run-1", help="wandb run name")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="batch size")
    parser.add_argument("--save-model-path", type=str, default="best_model.pt", help="path to save the best performing model")
    parser.add_argument("--steps-to-print", "--sp", type=int, default=100, help="steps between reporting to wandb")
    parser.add_argument("--embedding-dim", type=int, default=12, help="embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=24, help="hidden dimension")
    parser.add_argument("--ls", type=float, default=0.0, help="smoothing hyperparameter")
    parser.add_argument("--teacher-forcing-ratio", "--tfr", type=float, default=0.5, help="teacher forcing ratio")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Initializing model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EMB_DIM = args.embedding_dim
    HID_DIM = args.hidden_dim
    ploidy = 2

    enc = EncoderDiploid(args.num_parents, EMB_DIM, HID_DIM)
    dec = DecoderDiploidJoint(args.num_parents + 1, EMB_DIM, HID_DIM, ploidy=ploidy)
    model = Seq2SeqDiploidJoint(enc, dec, device, ploidy=ploidy)

    # Initializing training dataset
    training_filenames = os.listdir(args.training_data_path)
    training_dataset = LabeledDatasetDiploid(
        args.training_data_path,
        training_filenames,
        args.max_seq_length,
        args.num_parents,
        args.step_size
    )

    # Initializing validation dataset
    validation_filenames = os.listdir(args.validation_data_path)
    validation_dataset = LabeledDatasetDiploid(
        args.validation_data_path,
        validation_filenames,
        args.max_seq_length,
        args.num_parents,
        args.step_size
    )

    # Setting up optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = SmoothPredictDiploidJoint(lambda_smooth=args.ls)

    model.to(device)
    criterion.to(device)

    # start up wandb run
    wandb.init(
        project=args.project_name,
        entity="maize-genetics",
        name=args.run_name,
        config={
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "embedding_dim": EMB_DIM,
            "hidden_dim": HID_DIM,
            "teacher_forcing_ratio": args.teacher_forcing_ratio,
            "joint_pair_prediction": True,
            "lambda_smooth": args.ls
        }
    )

    best_loss = float('inf')

    # loop through epochs for training
    for epoch in range(args.num_epochs):
        dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

        # keep behavior close to your original script: no teacher forcing in training call by default
        # but this arg is available if you want to wire it in below.
        train_loss, train_acc = train(model, dataloader, optimizer, criterion, args.steps_to_print)
        test_loss, test_acc = evaluate(model, test_dataloader, criterion)

        wandb.log({
            "Epoch Loss": train_loss,
            "Epoch Accuracy": train_acc,
            "Test Loss": test_loss,
            "Test Accuracy": test_acc,
            "Epoch": epoch
        })

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), args.save_model_path)

    wandb.finish()


if __name__ == '__main__':
    main()