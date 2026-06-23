"""
E1 haploid training — GRITS-CRF on maize inbred data.

Maize inbred .npy files are [T, 25]: cols 0-23 = founder read counts, col 24 = label.
The CRF runs over K=24 founder states (nsw[i,j] = i!=j).
The encoder is architecturally identical to the diploid path; only the state layer differs.

Usage:
    pixi run --environment gpu python src/python/crf/train_haploid.py --workdir /workdir/esb33
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader

from python.crf.train_crf import FounderPathEncoder, NeuralCRF

NEG_INF = -1e4

INBRED_PREFIXES = [
    'B97', 'CML103', 'CML228', 'CML247', 'CML277', 'CML322', 'CML333',
    'CML52', 'CML69', 'HP301', 'Il14H', 'Ki11', 'Ki3', 'Ky21',
    'M162W', 'M37W', 'Mo18W', 'Ms71', 'NC350', 'NC358',
    'Oh43', 'Oh7B', 'P39', 'Tzi8',
]


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class LabeledDatasetHaploid(Dataset):
    """
    Reads 25-col inbred .npy matrices [T, 25]:
      cols 0:24  = founder read counts  (features)
      col  24    = founder label in [0, num_parents)  (-1 → num_parents = unknown)
    Windows into [window_size, num_parents] feature + [window_size] label tensors.
    """

    def __init__(self, file_dir, file_names, window_size=512, num_parents=24, step_size=128):
        self.file_dir = file_dir
        self.file_names = file_names
        self.window_size = window_size
        self.num_parents = num_parents
        self.step_size = step_size
        self.windows = self._build_windows()
        print(f"HaploidDataset: {len(self.windows)} windows from {len(file_names)} files")

    def _build_windows(self):
        windows = []
        for i, fname in enumerate(self.file_names):
            T = np.load(f"{self.file_dir}/{fname}", mmap_mode='r').shape[0]
            n = max(0, (T - self.window_size) // self.step_size)
            windows.extend((i, j) for j in range(n))
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        fi, wi = self.windows[idx]
        s = wi * self.step_size
        mat = np.load(
            f"{self.file_dir}/{self.file_names[fi]}",
            mmap_mode='r', allow_pickle=True
        )[s : s + self.window_size]

        features = torch.tensor(mat[:, :self.num_parents], dtype=torch.float)
        labels = torch.tensor(mat[:, self.num_parents].astype(np.int64), dtype=torch.long)
        labels[labels == -1] = self.num_parents  # unknown → last index
        return {"input_embeds": features, "labels": labels}


# --------------------------------------------------------------------------- #
#  Lightning module                                                             #
# --------------------------------------------------------------------------- #

class GRITSCRFHaploid(pl.LightningModule):
    """
    Haploid CRF: same FounderPathEncoder as the diploid model, but the CRF
    runs over K founder states with nsw[i,j] = (i != j).
    Diploid promotion (E4b) loads this encoder unchanged into the pair-state wrapper.
    """

    def __init__(self, num_parents=24, d_model=256, n_heads=8, n_layers=6,
                 lr=3e-4, weight_decay=1e-5, gate_reg=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.num_parents = num_parents
        self.lr = lr
        self.weight_decay = weight_decay
        self.gate_reg = gate_reg

        K = num_parents + 1  # +1 for unknown state
        self.encoder = FounderPathEncoder(d_model, n_heads, n_layers)
        self.crf = NeuralCRF()

        # nsw[i,j] = number of haplotype switches between state i and state j.
        # For haploid K states: 0 if i==j, 1 otherwise.
        nsw = (1.0 - torch.eye(K))
        self.register_buffer("nsw", nsw)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GRITSCRFHaploid: K={K} states, {n_params:,} params")

    def forward(self, X):
        B, T, K_feat = X.shape
        K = self.num_parents + 1
        X_pad = torch.cat([X, torch.zeros(B, T, 1, device=X.device)], dim=-1)  # [B,T,K]
        founder_mask = torch.ones(B, K, device=X.device)
        emis_f, g, c = self.encoder(X_pad, founder_mask)  # emis_f [B,T,K]
        tr = self.crf._trans(c, self.nsw)                 # [B,T,K,K]
        return emis_f, tr, g, c

    def _step(self, batch):
        X, tags = batch["input_embeds"], batch["labels"]
        emis_f, tr, g, c = self(X)
        crf_loss = self.crf.nll(emis_f, tr, tags)
        gate_loss = self.gate_reg * (1.0 - g).mean()
        return crf_loss + gate_loss, crf_loss, g, c, emis_f, tr

    def training_step(self, batch, batch_idx):
        loss, crf_loss, g, c, _, _ = self._step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/crf_loss", crf_loss)
        self.log("train/gate", g.mean())
        self.log("train/recomb", c.mean())
        self.log("train/stay_bonus", self.crf.stay_bonus)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, crf_loss, g, c, emis_f, tr = self._step(batch)
        tags = batch["labels"]
        pred = self.crf.viterbi(emis_f, tr)
        acc = (pred == tags).float().mean()
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/crf_loss", crf_loss)
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/gate", g.mean())
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=5)
        return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val/loss"}


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def list_inbred_files(data_dir, val_chr="10"):
    """Split maize inbred files into train / val by chromosome."""
    all_files = [
        f for f in os.listdir(data_dir)
        if f.endswith(".npy") and any(f.startswith(p + "_") for p in INBRED_PREFIXES)
    ]
    train = [f for f in all_files if f"_chr{val_chr}_" not in f]
    val   = [f for f in all_files if f"_chr{val_chr}_" in f]
    return sorted(train), sorted(val)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir",    default="/workdir/esb33")
    p.add_argument("--data-dir",   default="/workdir/smm477/ML-data/training")
    p.add_argument("--val-chr",    default="10",  help="Chromosome held out for val")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--window-size", type=int, default=512)
    p.add_argument("--step-size",   type=int, default=128)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--d-model",     type=int, default=256)
    p.add_argument("--n-heads",     type=int, default=8)
    p.add_argument("--n-layers",    type=int, default=6)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--gate-reg",    type=float, default=0.05)
    p.add_argument("--max-epochs",  type=int, default=50)
    p.add_argument("--patience",    type=int, default=10)
    args = p.parse_args()

    workdir = Path(args.workdir)
    ckpt_dir = workdir / "checkpoints" / "e1-haploid"
    log_dir  = workdir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_files, val_files = list_inbred_files(args.data_dir, args.val_chr)
    print(f"Train files: {len(train_files)}  |  Val files (chr{args.val_chr}): {len(val_files)}")

    train_ds = LabeledDatasetHaploid(args.data_dir, train_files,
                                     args.window_size, args.num_parents, args.step_size)
    val_ds   = LabeledDatasetHaploid(args.data_dir, val_files,
                                     args.window_size, args.num_parents, args.step_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = GRITSCRFHaploid(
        num_parents=args.num_parents,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        lr=args.lr,
        gate_reg=args.gate_reg,
    )

    callbacks = [
        ModelCheckpoint(dirpath=str(ckpt_dir), monitor="val/loss", mode="min",
                        save_top_k=3, filename="e1-{epoch:02d}-{val/loss:.3f}"),
        EarlyStopping(monitor="val/loss", mode="min", patience=args.patience),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=TensorBoardLogger(str(log_dir), name="e1-haploid"),
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
