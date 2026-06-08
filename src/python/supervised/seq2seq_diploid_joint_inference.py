import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import numpy as np
import math

# import your NEW joint model
from src.python.supervised.seq2seq_diploid_joint import (
    Seq2SeqDiploidJoint,
    EncoderDiploid,
    DecoderDiploidJoint
)


# =========================
# DATASET (UNCHANGED)
# =========================
class LabeledDatasetDiploidInference(Dataset):
    def __init__(self, file_dir, input_file_names, window_size=512, num_parents=24, step_size=128):
        self.file_dir = file_dir
        self.input_file_names = input_file_names
        self.window_size = window_size
        self.num_parents = num_parents
        self.step_size = step_size
        self.windows = self.__generate_windows__()
        self.n_windows = len(self.windows)

    def __generate_windows__(self):
        windows = []
        for idx in range(len(self.input_file_names)):
            filelen = np.load(f"{self.file_dir}/{self.input_file_names[idx]}").shape[0]
            num_windows = math.ceil(filelen / self.step_size)
            windows.extend([(idx, idy) for idy in range(num_windows)])
        return windows

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        file_idx, pos_idx = self.windows[idx]

        pos_start = pos_idx * self.step_size
        pos_end = pos_start + self.window_size

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

        if ip.shape[1] == 25:
            matrix = np.concatenate([ip, ip[:, self.num_parents:self.num_parents+1]], axis=1)
        elif ip.shape[1] == 26:
            matrix = ip[:, 0:self.num_parents+2]
        else:
            raise RuntimeError("Input matrix is wrong shape")

        labels = torch.tensor(matrix[:, self.num_parents:self.num_parents+2], dtype=torch.int64)
        labels[labels == -1] = 24

        return {
            "input_embeds": torch.tensor(matrix[:, :-2], dtype=torch.float),
            "labels": labels
        }


# =========================
# JOINT DECODING
# =========================
def decode_joint_logits(logits, force_predictions=False):
    """
    logits: (B, W, V, V)
    returns: (B, W, 2)
    """
    B, W, V, _ = logits.shape

    if force_predictions:
        logits = logits[:, :, :-1, :-1]  # remove "unknown" class

    logits_flat = logits.reshape(B, W, -1)  # (B, W, V*V)
    pair_idx = torch.argmax(logits_flat, dim=-1)  # (B, W)

    V_eff = logits.shape[2]

    p1 = pair_idx // V_eff
    p2 = pair_idx % V_eff

    return torch.stack([p1, p2], dim=-1)  # (B, W, 2)


# =========================
# SLIDING WINDOW INFERENCE
# =========================
def inference_sliding(model, iterator, force_predictions=False):
    device = model.device
    predictions = []
    model.eval()

    num_batches = len(iterator)

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(iterator, desc="Evaluating...")):
            input_embeds = batch["input_embeds"].to(device)

            # [B, W, V, V]
            batch_predictions = model(input_embeds)

            W = batch_predictions.shape[1]
            start = W // 4
            end = (W // 4) * 3

            # decode joint logits → diploid pairs
            arg = decode_joint_logits(batch_predictions, force_predictions)

            if batch_num == 0:
                first = arg[0, :end]
                mid = arg[1:, start:end]

                predictions.append(first.cpu().reshape(-1, 2))
                if mid.numel() > 0:
                    predictions.append(mid.cpu().reshape(-1, 2))

            elif batch_num == num_batches - 1:
                mid = arg[:-1, start:end]
                last = arg[-1, start:]

                if mid.numel() > 0:
                    predictions.append(mid.cpu().reshape(-1, 2))
                predictions.append(last.cpu().reshape(-1, 2))

            else:
                mid = arg[:, start:end]
                predictions.append(mid.cpu().reshape(-1, 2))

    preds = torch.cat(predictions, dim=0).to(torch.int64)
    return preds.numpy()


# =========================
# ARGS
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--batch-size", "-b", type=int, default=8)
    parser.add_argument("--model-path", type=str, default="best_model.pt")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--num-parents", "--np", type=int, default=24)
    parser.add_argument("--max-seq-length", "--sl", type=int, default=512)
    parser.add_argument("--step-size", "-s", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--force-preds", action="store_true", help="forces predictions / disables unknown option")

    return parser.parse_args()


# =========================
# MAIN
# =========================
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    ckpt = torch.load(args.model_path, map_location=device)

    enc = EncoderDiploid(args.num_parents, args.embedding_dim, args.hidden_dim)
    dec = DecoderDiploidJoint(args.num_parents + 1, args.embedding_dim, args.hidden_dim, ploidy=2)
    model = Seq2SeqDiploidJoint(enc, dec, device, ploidy=2)

    model.load_state_dict(ckpt)
    model.to(device)

    os.makedirs(args.save_dir, exist_ok=True)

    filenames = os.listdir(args.data_path)

    for file in filenames:
        dataset = LabeledDatasetDiploidInference(
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