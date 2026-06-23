"""
GRITS-CRF Training Script with PyTorch Lightning
Diploid founder path imputation using neural CRF
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

NEG_INF = -1e4


# --------------------------------------------------------------------------- #
#  Dataset with diploid pair conversion                                        #
# --------------------------------------------------------------------------- #

# Create a shared function for consistent pair indexing
def create_diploid_pairs(num_parents):
    """Create consistent pair mapping used by both dataset and model"""
    pairs = []
    pair_to_idx = {}

    K = num_parents + 1  # +1 for unknown state
    for i in range(K):
        for j in range(i, K):
            idx = len(pairs)
            pairs.append((i, j))
            pair_to_idx[(i, j)] = idx
            if i != j:
                pair_to_idx[(j, i)] = idx

    return pairs, pair_to_idx

class LabeledDatasetDiploid(Dataset):
    def __init__(self, file_dir, input_file_names, window_size=512, num_parents=24, step_size=128):
        self.file_dir = file_dir
        self.input_file_names = input_file_names
        self.window_size = window_size
        self.num_parents = num_parents
        self.step_size = step_size

        # Precompute pair indices for diploid states
        # self.pair_idx, self.pairs = self._create_pair_index()
        # self.windows = self.__generate_windows__()
        # self.n_windows = len(self.windows)

        # Use shared function
        self.pairs, self.pair_idx = create_diploid_pairs(num_parents)
        print(f"Created {len(self.pairs)} pair states for {num_parents} parents")

        self.windows = self.__generate_windows__()
        self.n_windows = len(self.windows)

        print(f"Dataset created: {self.n_windows} windows, {len(self.pairs)} pair states")

    # def _create_pair_index(self):
    #     """Create mapping from unordered founder pairs to indices"""
    #     pairs = []
    #     pair_to_idx = {}
    #
    #     # Include all unordered pairs {i,j} where i <= j
    #     for i in range(self.num_parents + 1):  # +1 for unknown state
    #         for j in range(i, self.num_parents + 1):
    #             idx = len(pairs)
    #             pairs.append((i, j))
    #             pair_to_idx[(i, j)] = idx
    #             if i != j:  # Also map {j,i} to same index for unordered pairs
    #                 pair_to_idx[(j, i)] = idx
    #
    #     return pair_to_idx, pairs

    def _diploid_to_pair_idx(self, diploid_labels):
        """Convert [T, 2] diploid labels to [T] pair indices"""
        """Convert [T, 2] diploid labels to [T] pair indices with bounds checking"""
        T = diploid_labels.shape[0]
        pair_indices = torch.zeros(T, dtype=torch.long)

        for t in range(T):
            i, j = diploid_labels[t, 0].item(), diploid_labels[t, 1].item()

            # Add bounds checking
            if (i, j) not in self.pair_idx:
                print(f"Warning: Invalid pair ({i}, {j}) at position {t}")
                # Fallback to unknown state pair
                unknown_idx = self.num_parents
                pair_indices[t] = self.pair_idx[(unknown_idx, unknown_idx)]
            else:
                pair_indices[t] = self.pair_idx[(i, j)]

        return pair_indices
        # T = diploid_labels.shape[0]
        # pair_indices = torch.zeros(T, dtype=torch.long)
        #
        # for t in range(T):
        #     i, j = diploid_labels[t, 0].item(), diploid_labels[t, 1].item()
        #     # Look up the pair index for this unordered pair
        #     pair_indices[t] = self.pair_idx[(i, j)]
        #
        # return pair_indices

    def __generate_windows__(self):
        windows = []
        for idx in range(len(self.input_file_names)):
            filelen = np.load(f"{self.file_dir}/{self.input_file_names[idx]}").shape[0]
            num_windows = (filelen - self.window_size) // self.step_size
            windows.extend([(idx, idy) for idy in range(num_windows)])
        return windows

    def __len__(self):
        return self.n_windows

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

        if ip.shape[1] != self.num_parents + 2:  # copy haploid labels
            matrix = np.concatenate([ip, ip[:, self.num_parents:self.num_parents + 1]], axis=1)
        else:  # diploid labels
            matrix = ip[:, 0:self.num_parents + 2]

        # Convert labels and handle unknown states
        labels = torch.tensor(matrix[:, self.num_parents:self.num_parents + 2], dtype=torch.int64)
        labels[labels == -1] = self.num_parents

        # Convert diploid labels to pair indices
        pair_labels = self._diploid_to_pair_idx(labels)

        return {
            "input_embeds": torch.tensor(matrix[:, :-2], dtype=torch.float),
            "labels": pair_labels,
            "diploid_labels": labels  # Keep for debugging
        }


# --------------------------------------------------------------------------- #
#  Model components (from original encoder_crf.py)                             #
# --------------------------------------------------------------------------- #

class FounderPathEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=4, ext_dim=0,
                 time_local_emis=False):
        super().__init__()
        self.d_model = d_model
        self.time_local_emis = time_local_emis
        self.cell = nn.Sequential(nn.Linear(1, d_model), nn.GELU(),
                                  nn.Linear(d_model, d_model))
        self.ext = nn.Linear(ext_dim, d_model) if ext_dim > 0 else None
        self.fpool = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fquery = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                         batch_first=True, activation="gelu")
        self.pos_encoder = nn.TransformerEncoder(enc, n_layers)
        self.gate_head = nn.Linear(d_model, 1)
        self.recomb_head = nn.Linear(d_model + 3, 1)
        self.scale = d_model ** -0.5

    @staticmethod
    def _posenc(T, d, device):
        pe = torch.zeros(T, d, device=device)
        pos = torch.arange(T, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2, device=device).float()
                        * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, X, founder_mask, dbp=None, ext_emb=None):
        B, T, K = X.shape
        cells = self.cell(torch.log1p(X).unsqueeze(-1))

        cf = cells.reshape(B * T, K, self.d_model)
        q = self.fquery.expand(B * T, 1, self.d_model)
        kpad = ~founder_mask.bool().unsqueeze(1).expand(B, T, K).reshape(B * T, K)
        h, _ = self.fpool(q, cf, cf, key_padding_mask=kpad)
        h = h.reshape(B, T, self.d_model) + self._posenc(T, self.d_model, X.device)
        H = self.pos_encoder(h)

        if self.time_local_emis:
            # Per-site founder key: emission at site t scores founder f using
            # its cell embedding AT t, not averaged over the window. Required
            # when the active founder switches within a window (recombination).
            if self.ext is not None and ext_emb is not None:
                raise NotImplementedError("ext_emb unsupported with time_local_emis")
            cf_local = cells.masked_fill(~founder_mask.bool().view(B, 1, K, 1), 0.0)
            emis = torch.einsum("btd,btkd->btk", H, cf_local) * self.scale
        else:
            e = cells.mean(dim=1)
            if self.ext is not None and ext_emb is not None:
                e = e + self.ext(ext_emb)
            e = e.masked_fill(~founder_mask.bool().unsqueeze(-1), 0.0)
            emis = torch.einsum("btd,bkd->btk", H, e) * self.scale
        emis = emis.masked_fill(~founder_mask.bool().unsqueeze(1), NEG_INF)

        g = torch.sigmoid(self.gate_head(H)).squeeze(-1)
        valid = founder_mask.bool().unsqueeze(1)
        emis = torch.where(valid, g.unsqueeze(-1) * emis.clamp(min=NEG_INF / 2),
                           torch.full_like(emis, NEG_INF))

        depth = torch.log1p(X.sum(-1, keepdim=True))
        ent = self._entropy(emis, founder_mask).unsqueeze(-1)
        if dbp is None:
            dbp = torch.ones(B, T, 1, device=X.device)
        feats = torch.cat([H, depth, ent, torch.log1p(dbp)], dim=-1)
        c = F.softplus(self.recomb_head(feats)).squeeze(-1)
        return emis, g, c

    def _entropy(self, emis, founder_mask):
        p = torch.softmax(emis.masked_fill(~founder_mask.bool().unsqueeze(1), NEG_INF), dim=-1)
        return -(p.clamp_min(1e-9).log() * p).sum(-1)


class NeuralCRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.stay_bonus = nn.Parameter(torch.tensor(2.0))

    def _trans(self, c, nsw):
        B, T = c.shape
        tr = -c[:, :, None, None] * nsw[None, None]
        tr = tr + (nsw == 0).float()[None, None] * self.stay_bonus
        return tr

    def partition(self, emis, tr):
        B, T, S = emis.shape
        a = emis[:, 0]
        for t in range(1, T):
            a = emis[:, t] + torch.logsumexp(a.unsqueeze(2) + tr[:, t], dim=1)
        return torch.logsumexp(a, dim=1)

    def score(self, emis, tr, tags):
        B, T, S = emis.shape
        bi = torch.arange(B, device=emis.device)
        s = emis[bi, 0, tags[:, 0]]
        for t in range(1, T):
            s = s + tr[bi, t, tags[:, t - 1], tags[:, t]] + emis[bi, t, tags[:, t]]
        return s

    def nll(self, emis, tr, tags):
        return (self.partition(emis, tr) - self.score(emis, tr, tags)).mean()

    @torch.no_grad()
    def viterbi(self, emis, tr):
        B, T, S = emis.shape
        delta = emis[:, 0]
        bp = []
        for t in range(1, T):
            sc = delta.unsqueeze(2) + tr[:, t]
            best, idx = sc.max(dim=1)
            delta = emis[:, t] + best
            bp.append(idx)
        path = [delta.argmax(dim=1)]
        for idx in reversed(bp):
            path.append(idx.gather(1, path[-1].unsqueeze(1)).squeeze(1))
        return torch.stack(path[::-1], dim=1)


# --------------------------------------------------------------------------- #
#  Diploid utilities                                                            #
# --------------------------------------------------------------------------- #

def diploid_pair_index(K, device="cpu"):
    pairs = [(i, j) for i in range(K) for j in range(i, K)]
    idx = torch.tensor(pairs, device=device)
    P = len(pairs)
    nsw = torch.zeros(P, P, device=device)
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            nsw[a, b] = min((i != k) + (j != l), (i != l) + (j != k))
    return idx, nsw


# def diploid_emissions(emis_f, pair_idx):
#     i, j = pair_idx[:, 0], pair_idx[:, 1]
#     return emis_f[..., i] + emis_f[..., j]


def diploid_emissions(emis_f, pair_idx):
    """
    Convert founder emissions [B,T,K] to pair emissions [B,T,P]
    """
    B, T, K = emis_f.shape
    P = pair_idx.shape[0]

    # Extract founder indices
    i_idx = pair_idx[:, 0]  # [P]
    j_idx = pair_idx[:, 1]  # [P]

    # Check bounds
    if i_idx.max() >= K or j_idx.max() >= K:
        raise ValueError(f"Pair indices exceed founder count: max_i={i_idx.max()}, max_j={j_idx.max()}, K={K}")

    # Vectorized indexing: [B, T, P]
    emis_i = emis_f[:, :, i_idx]  # [B, T, P]
    emis_j = emis_f[:, :, j_idx]  # [B, T, P]

    return emis_i + emis_j


# --------------------------------------------------------------------------- #
#  Lightning Module                                                             #
# --------------------------------------------------------------------------- #

class GRITSCRFModel(pl.LightningModule):
    def __init__(self, num_parents=24, d_model=128, n_heads=4, n_layers=4,
                 lr=3e-4, weight_decay=1e-5, gate_reg=0.05):
        super().__init__()
        self.save_hyperparameters()

        self.num_parents = num_parents
        self.lr = lr
        self.weight_decay = weight_decay
        self.gate_reg = gate_reg

        # Model components
        self.encoder = FounderPathEncoder(d_model, n_heads, n_layers)
        self.crf = NeuralCRF()

        # Diploid state space
        K_total = num_parents + 1  # +1 for unknown state
        pairs, _ = create_diploid_pairs(num_parents)

        # Create diploid index tensors
        pair_idx_list = []
        for i, j in pairs:
            pair_idx_list.append([i, j])

        self.register_buffer("pair_idx", torch.tensor(pair_idx_list))
        # self.register_buffer("pair_idx", diploid_pair_index(K)[0])
        P = len(pairs)
        nsw = torch.zeros(P, P)
        for a, (i, j) in enumerate(pairs):
            for b, (k, l) in enumerate(pairs):
                nsw[a, b] = min((i != k) + (j != l), (i != l) + (j != k))
        self.register_buffer("nsw", nsw)

        print(f"Model: {P} pair states, pair_idx shape: {self.pair_idx.shape}")
        # self.register_buffer("nsw", diploid_pair_index(K)[1])

        # Metrics
        self.train_losses = []
        self.val_losses = []

    def forward(self, X):
        B, T, K = X.shape

        # Pad input to include unknown state (index 24)
        X_padded = torch.cat([
            X,
            torch.zeros(B, T, 1, device=X.device, dtype=X.dtype)
        ], dim=-1)  # Now [B, T, 25]

        # founder_mask = torch.ones(B, K, device=X.device)
        founder_mask = torch.ones(B, K+1, device=X.device)
        # emis_f, g, c = self.encoder(X, founder_mask)
        emis_f, g, c = self.encoder(X_padded, founder_mask)

        # Debug: Check shapes before diploid_emissions
        print(f"Debug: emis_f shape: {emis_f.shape}, pair_idx shape: {self.pair_idx.shape}")
        print(f"Debug: pair_idx max values: i={self.pair_idx[:, 0].max()}, j={self.pair_idx[:, 1].max()}")

        emis_p = diploid_emissions(emis_f, self.pair_idx)
        tr = self.crf._trans(c, self.nsw)
        return emis_p, tr, g, c

    def training_step(self, batch, batch_idx):
        X = batch["input_embeds"]
        tags = batch["labels"]

        emis_p, tr, g, c = self(X)

        B, T, K = X.shape
        max_pair_idx = self.pair_idx.shape[0] - 1

        if tags.max() > max_pair_idx:
            print(f"ERROR: Label index {tags.max()} exceeds max pair index {max_pair_idx}")
            print(f"Batch shape: {X.shape}, Tags shape: {tags.shape}")
            print(f"Unique labels: {tags.unique()}")
            return None

        try:
            emis_p, tr, g, c = self(X)
            crf_loss = self.crf.nll(emis_p, tr, tags)
            gate_loss = self.gate_reg * (1.0 - g).mean()
            total_loss = crf_loss + gate_loss

            if batch_idx % 50 == 0:
                # Check if parameters are actually updating
                total_grad_norm = 0
                for p in self.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.norm().item() ** 2
                total_grad_norm = total_grad_norm ** 0.5

                self.log("train/grad_norm", total_grad_norm)

                if total_grad_norm < 1e-6:
                    print("WARNING: Very small gradients - model may not be learning")

            # Logging
            self.log("train/crf_loss", crf_loss, prog_bar=True)
            self.log("train/gate_loss", gate_loss)
            self.log("train/total_loss", total_loss)
            self.log("train/mean_gate", g.mean())
            self.log("train/mean_recomb", c.mean())
            self.log("train/stay_bonus", self.crf.stay_bonus)
            return total_loss
        except RuntimeError as e:
            print(f"Runtime error in training step:")
            print(f"X shape: {X.shape}")
            print(f"tags shape: {tags.shape}, min: {tags.min()}, max: {tags.max()}")
            print(f"emis_p shape: {emis_p.shape if 'emis_p' in locals() else 'not created'}")
            raise e

        # # CRF loss
        # crf_loss = self.crf.nll(emis_p, tr, tags)
        #
        # # Gate regularization (keep gate near 1 by default)
        # gate_loss = self.gate_reg * (1.0 - g).mean()
        #
        # total_loss = crf_loss + gate_loss
        #
        # # Logging
        # self.log("train/crf_loss", crf_loss, prog_bar=True)
        # self.log("train/gate_loss", gate_loss)
        # self.log("train/total_loss", total_loss)
        # self.log("train/mean_gate", g.mean())
        # self.log("train/mean_recomb", c.mean())
        # self.log("train/stay_bonus", self.crf.stay_bonus)
        #
        # return total_loss

    def validation_step(self, batch, batch_idx):
        X = batch["input_embeds"]
        tags = batch["labels"]

        emis_p, tr, g, c = self(X)
        crf_loss = self.crf.nll(emis_p, tr, tags)
        gate_loss = self.gate_reg * (1.0 - g).mean()
        total_loss = crf_loss + gate_loss

        if batch_idx == 0:  # Log first batch details each epoch
            # Check if model is actually changing predictions
            with torch.no_grad():
                pred_paths = self.crf.viterbi(emis_p, tr)

            # Log prediction diversity
            unique_predictions = [len(torch.unique(pred_paths[i])) for i in range(min(4, pred_paths.shape[0]))]
            self.log("val/pred_diversity", sum(unique_predictions) / len(unique_predictions))

            # Log label diversity for comparison
            unique_labels = [len(torch.unique(tags[i])) for i in range(min(4, tags.shape[0]))]
            self.log("val/label_diversity", sum(unique_labels) / len(unique_labels))

            # Are predictions just stuck on one state?
            mode_predictions = [torch.mode(pred_paths[i])[0].item() for i in range(min(4, pred_paths.shape[0]))]
            print(f"Dominant predicted states: {mode_predictions}")

        # Path accuracy
        pred_paths = self.crf.viterbi(emis_p, tr)
        path_acc = (pred_paths == tags).float().mean()

        print(str(tags[0:10]))

        self.log("val/crf_loss", crf_loss, prog_bar=True)
        self.log("val/total_loss", total_loss)
        self.log("val/path_acc", path_acc, prog_bar=True)
        self.log("val/mean_gate", g.mean())

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
            # Remove verbose=True - it was deprecated
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/total_loss"
        }


# --------------------------------------------------------------------------- #
#  Training script                                                              #
# --------------------------------------------------------------------------- #

def main():
    # Configuration
    config = {
        # "data_dir": "/path/to/your/data",  # UPDATE THIS
        "data_dir": "/workdir/smm477/ML-data/training",  # UPDATE THIS
        # "train_files": ["train_file1.npy", "train_file2.npy"],  # UPDATE THIS
        "train_files": ["B97_chr10_matrix_data-1_0.1x.npy", "B97_chr9_matrix_data-1_1x.npy"],  # UPDATE THIS
        # "val_files": ["val_file1.npy"],  # UPDATE THIS
        "val_files": ["B97_chr6_matrix_data-1_1x.npy"],  # UPDATE THIS
        "num_parents": 24,
        "window_size": 512,
        "step_size": 128,
        "batch_size": 8,
        "num_workers": 4,
        # "d_model": 128,
        "d_model": 64,
        "n_heads": 4,
        # "n_layers": 4,
        "n_layers": 2,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "gate_reg": 0.05,
        "max_epochs": 100,
        "patience": 10
    }

    # Datasets
    train_dataset = LabeledDatasetDiploid(
        config["data_dir"],
        config["train_files"],
        window_size=config["window_size"],
        num_parents=config["num_parents"],
        step_size=config["step_size"]
    )

    val_dataset = LabeledDatasetDiploid(
        config["data_dir"],
        config["val_files"],
        window_size=config["window_size"],
        num_parents=config["num_parents"],
        step_size=config["step_size"]
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # In your main() function, before training:
    print("Checking dataset...")
    sample = train_dataset[0]
    print(f"Input shape: {sample['input_embeds'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Label range: {sample['labels'].min()} to {sample['labels'].max()}")
    print(f"Unique labels: {sample['labels'].unique()}")

    # Check a batch
    sample_batch = next(iter(train_loader))
    print(f"Batch input shape: {sample_batch['input_embeds'].shape}")
    print(f"Batch labels shape: {sample_batch['labels'].shape}")
    print(f"Batch label range: {sample_batch['labels'].min()} to {sample_batch['labels'].max()}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # Model
    model = GRITSCRFModel(
        num_parents=config["num_parents"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        gate_reg=config["gate_reg"]
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/total_loss",
        mode="min",
        save_top_k=3,
        filename="grits-crf-{epoch:02d}-{val_total_loss:.3f}"
    )

    early_stop_callback = EarlyStopping(
        monitor="val/total_loss",
        mode="min",
        patience=config["patience"],
        verbose=True
    )

    # Logger
    logger = TensorBoardLogger("logs", name="grits_crf")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator="auto",
        devices="auto",
        precision=16,  # Mixed precision for speed
        gradient_clip_val=1.0
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()