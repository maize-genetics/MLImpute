import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import numpy as np
import math

# import your NEW joint model
from supervised.seq2seq_diploid_joint_bert import (
    Seq2SeqDiploidJoint,
    EncoderDiploid,
    DecoderDiploidJoint
)

from supervised.seq2seq_diploid_joint_inference import (
    decode_joint_logits, inference_sliding, parse_args)


# =========================
# DATASET (UNCHANGED)
# =========================
class LabeledDatasetDiploidPosInference(Dataset):
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
        windows = []
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


# =========================
# MAIN
# =========================
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    ckpt = torch.load(args.model_path, map_location=device)

    enc = EncoderDiploid(args.num_parents + 1, args.embedding_dim, args.max_seq_length)
    dec = DecoderDiploidJoint(args.num_parents + 1, args.embedding_dim, args.hidden_dim, ploidy=2)
    model = Seq2SeqDiploidJoint(enc, dec, device, ploidy=2)

    model.load_state_dict(ckpt)
    model.to(device)

    os.makedirs(args.save_dir, exist_ok=True)

    filenames = os.listdir(args.data_path)

    for file in filenames:
        dataset = LabeledDatasetDiploidPosInference(
            args.data_path,
            [file],
            args.max_seq_length,
            args.num_parents,
            args.step_size
        )

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        predictions = inference_sliding(model, dataloader, args.force_preds)
        predictions = predictions.reshape(-1, 2)

        pred_file = file.split("_matrix")[0]

        ps4g_len = len(np.load(f"{args.data_path}/{file}", allow_pickle=True))
        assert predictions.shape[0] >= ps4g_len

        np.save(os.path.join(args.save_dir, pred_file), predictions[:ps4g_len, :])


if __name__ == "__main__":
    main()