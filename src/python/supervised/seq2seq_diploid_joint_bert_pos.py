"""
This version expects input data with positional embeddings.
Data format: [read_count_1, read_count_2, ..., read_count_N, position, label_1, label_2]
Use --positional-embed flag when generating training data.
"""

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
from transformers import ModernBertModel, ModernBertConfig
from src.python.supervised.seq2seq_diploid_joint_bert import (
    path_acc_diploid, decode_joint_predictions, EncoderDiploid,
    DecoderDiploidJoint, Seq2SeqDiploidJoint, JointPairCrossEntropyLoss,
    SmoothPredictDiploidJoint, train, evaluate, parse_args)


# Dataset class
# there will be multiple input files to be memory-mapped with numpy
# and the labels are a part of the regular input files, not a separate window
class LabeledDatasetDiploidPos(Dataset):
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

        # Now expecting: [position, features..., label1, label2]
        expected_diploid = self.num_parents + 3  # 24 + 1 + 2 = 27
        expected_haploid = self.num_parents + 2  # 24 + 1 + 1 = 26

        if ip.shape[1] == expected_diploid:
            matrix = ip
        elif ip.shape[1] == expected_haploid:
            # Duplicate single label for diploid format
            matrix = np.concatenate([ip, ip[:, -1:]], axis=1)
        else:
            raise ValueError(f"Expected {expected_diploid} or {expected_haploid} columns, got {ip.shape[1]}")

        # Extract labels (last 2 columns)
        labels = torch.tensor(matrix[:, -2:], dtype=torch.int64)
        labels[labels == -1] = self.num_parents

        return {
            "input_embeds": torch.tensor(matrix[:, :-2], dtype=torch.float),  # All except last 2
            "labels": labels
        }


def main():
    args = parse_args()

    # Initializing model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EMB_DIM = args.embedding_dim
    HID_DIM = args.hidden_dim
    ploidy = 2

    enc = EncoderDiploid(args.num_parents + 1, EMB_DIM, args.max_seq_length)
    dec = DecoderDiploidJoint(args.num_parents + 1, EMB_DIM, HID_DIM, ploidy=ploidy)
    model = Seq2SeqDiploidJoint(enc, dec, device, ploidy=ploidy)

    # Initializing training dataset
    training_filenames = os.listdir(args.training_data_path)
    training_dataset = LabeledDatasetDiploidPos(
        args.training_data_path,
        training_filenames,
        args.max_seq_length,
        args.num_parents,
        args.step_size
    )

    # Initializing validation dataset
    validation_filenames = os.listdir(args.validation_data_path)
    validation_dataset = LabeledDatasetDiploidPos(
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