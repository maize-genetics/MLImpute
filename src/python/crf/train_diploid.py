"""
E4 diploid training — joint pair-state CRF on the shared FounderPathEncoder.

The encoder is unchanged (one emis_f [B,T,K] per founder). The diploid layer
forms pair-state emissions emis_p[b,t,(i,j)] = emis_f[i] + emis_f[j] over the
P = K(K+1)/2 unordered founder pairs, and decodes a pair path with a CRF whose
transition cost is -c * (#chromosomes that switch), nsw in {0,1,2}. A two-
chromosome switch therefore costs exp(-2c) = exp(-c)^2 — two INDEPENDENT
chromosome switches, matching the generative sim (no hard ban).

Reuses the state-count-agnostic forward/Viterbi structure from train_haploid
with a pair switch matrix. Only [B,P,P] is materialized per timestep (never
[B,T,P,P]).

Data: (N, T, K+2) — cols 0:K features, col K = H1, col K+1 = H2 (make_splits).

Usage:
    pixi run --environment gpu python src/python/crf/train_diploid.py \
        --data /workdir/esb33/data/training/sim_diploid_512.npy \
        --time-local-emis --lr 1e-4 --warmup-steps 500 --precision bf16-mixed \
        --max-epochs 5 --run-name diploid-pair
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

torch.set_float32_matmul_precision("medium")
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader

from python.crf.train_crf import FounderPathEncoder
from python.crf.train_haploid import make_splits

# --------------------------------------------------------------------------- #
#  Pair-state CRF — state-count-agnostic recursion with a pair switch matrix.  #
#  Transition cost -c*nsw, nsw in {0,1,2}: a two-chromosome switch costs        #
#  exp(-2c) = exp(-c)^2, i.e. two INDEPENDENT chromosome switches (matches the  #
#  generative sim). No hard ban — simultaneous switches are rare, not illegal.  #
# --------------------------------------------------------------------------- #

@torch.jit.script
def _dcrf_nll(emis: torch.Tensor, c: torch.Tensor, nsw: torch.Tensor,
              stay_bonus: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
    """Pair-state CRF NLL. emis [B,T,P], c [B,T], nsw/stay [P,P], tags [B,T]."""
    B, T, P = emis.shape
    stay_mask = (nsw == 0).float()
    a = emis[:, 0]
    for t in range(1, T):
        tr_t = -c[:, t, None, None] * nsw[None] + stay_bonus * stay_mask[None]
        a = emis[:, t] + torch.logsumexp(a.unsqueeze(2) + tr_t, dim=1)
    log_Z = torch.logsumexp(a, dim=1)

    bi = torch.arange(B, device=emis.device)
    t_idx = torch.arange(T, device=emis.device)
    emis_score = emis[bi.unsqueeze(1), t_idx.unsqueeze(0), tags].sum(1)
    # transition score along the true path: -c*nsw[prev,next] + stay where equal
    prev = tags[:, :-1]
    nxt = tags[:, 1:]
    nsw_path = nsw[prev, nxt]                                   # [B,T-1]
    stay_path = (nsw_path == 0).float()
    tr_score = (-c[:, 1:] * nsw_path + stay_bonus * stay_path).sum(1)
    return (log_Z - emis_score - tr_score).mean()


@torch.jit.script
def _dcrf_viterbi(emis: torch.Tensor, c: torch.Tensor, nsw: torch.Tensor,
                  stay_bonus: torch.Tensor) -> torch.Tensor:
    """Pair-state Viterbi. emis [B,T,P] -> [B,T] pair indices."""
    B, T, P = emis.shape
    stay_mask = (nsw == 0).float()
    delta = emis[:, 0]
    bp = torch.zeros(T - 1, B, P, dtype=torch.long, device=emis.device)
    for t in range(1, T):
        tr_t = -c[:, t, None, None] * nsw[None] + stay_bonus * stay_mask[None]
        sc = delta.unsqueeze(2) + tr_t
        best, idx = sc.max(dim=1)
        delta = emis[:, t] + best
        bp[t - 1] = idx
    path = torch.zeros(B, T, dtype=torch.long, device=emis.device)
    path[:, T - 1] = delta.argmax(dim=1)
    for t in range(T - 2, -1, -1):
        path[:, t] = bp[t].gather(1, path[:, t + 1].unsqueeze(1)).squeeze(1)
    return path


def build_pair_tables(K):
    """Unordered founder pairs over K states. Returns:
      pi, pj      [P]      sorted member indices of each pair (i<=j)
      pair_table  [K,K]    (a,b) -> pair index (order-insensitive)
      nsw_pair    [P,P]    min #chromosome switches between pairs (0,1,2)
    """
    pairs = [(i, j) for i in range(K) for j in range(i, K)]
    P = len(pairs)
    idx = {p: k for k, p in enumerate(pairs)}
    pi = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    pj = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    pair_table = torch.zeros(K, K, dtype=torch.long)
    for a in range(K):
        for b in range(K):
            pair_table[a, b] = idx[(min(a, b), max(a, b))]

    nsw = torch.zeros(P, P, dtype=torch.float32)
    for p, (a, b) in enumerate(pairs):
        for q, (c, d) in enumerate(pairs):
            s = min((a != c) + (b != d), (a != d) + (b != c))
            nsw[p, q] = float(s)
    return pi, pj, pair_table, nsw


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class PreWindowedDiploidDataset(Dataset):
    """(N,T,K+2): cols 0:K features, col K = H1, col K+1 = H2. Returns the
    feature window plus the two haplotype labels (pair index built in the
    module to keep the K×K table in one place)."""
    def __init__(self, data, num_parents=24):
        self.data = data
        self.K = num_parents

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        feats = torch.tensor(row[:, :self.K], dtype=torch.float32)
        h1 = np.clip(row[:, self.K].astype(np.int64), 0, self.K)
        h2 = np.clip(row[:, self.K + 1].astype(np.int64), 0, self.K)
        return {"input_embeds": feats,
                "h1": torch.tensor(h1, dtype=torch.long),
                "h2": torch.tensor(h2, dtype=torch.long)}


def make_diploid_splits(path, num_parents, val_frac, test_frac, limit_n=0):
    """Same deterministic head-slice split as make_splits, diploid dataset."""
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    if limit_n and limit_n < len(data):
        data = data[:limit_n]
    N = len(data)
    n_test = int(N * test_frac)
    n_val = int(N * val_frac)
    n_tr = N - n_val - n_test
    mk = lambda a: PreWindowedDiploidDataset(a, num_parents)
    print(f"Diploid {Path(path).name}: N={N:,} cols={data.shape[-1]}  "
          f"train={n_tr:,} val={n_val:,} test={n_test:,}")
    return mk(data[:n_tr]), mk(data[n_tr:n_tr + n_val]), mk(data[n_tr + n_val:])


# --------------------------------------------------------------------------- #
#  Lightning module                                                            #
# --------------------------------------------------------------------------- #

class GRITSCRFDiploid(pl.LightningModule):
    def __init__(self, num_parents=24, d_model=256, n_heads=8, n_layers=6,
                 lr=1e-4, weight_decay=1e-5, gate_reg=0.05, time_local_emis=False,
                 warmup_steps=0, homo_penalty=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.num_parents = num_parents
        self.lr = lr
        self.weight_decay = weight_decay
        self.gate_reg = gate_reg
        self.warmup_steps = warmup_steps
        self.homo_penalty = homo_penalty

        K = num_parents + 1                          # +1 unknown, matches encoder
        self.encoder = FounderPathEncoder(d_model, n_heads, n_layers,
                                          time_local_emis=time_local_emis)
        self.stay_bonus = nn.Parameter(torch.tensor(2.0))

        pi, pj, pair_table, nsw = build_pair_tables(K)
        self.register_buffer("pi", pi)
        self.register_buffer("pj", pj)
        self.register_buffer("pair_table", pair_table)
        self.register_buffer("nsw_pair", nsw)
        # Het prior: with one read/site the emission emis_f[i]+emis_f[j] is
        # maximized by the homozygous pair of the observed founder, so without a
        # counter-force the decode collapses to all-homozygous. Subtract a
        # constant from homozygous pair-states (i==j), matching diploid_hmm.
        self.register_buffer("homo_mask", (pi == pj).float())
        self.P = pi.numel()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GRITSCRFDiploid: K={K} states, P={self.P} pair-states, "
              f"{n_params:,} params")

    def forward(self, X):
        B, T, K_feat = X.shape
        K = self.num_parents + 1
        X_pad = torch.cat([X, torch.zeros(B, T, 1, device=X.device)], dim=-1)
        founder_mask = torch.ones(B, K, device=X.device)
        emis_f, g, c = self.encoder(X_pad, founder_mask)         # [B,T,K]
        emis_p = emis_f[..., self.pi] + emis_f[..., self.pj]     # [B,T,P]
        if self.homo_penalty != 0.0:
            emis_p = emis_p - self.homo_penalty * self.homo_mask
        return emis_p, g, c

    def _pair_labels(self, h1, h2):
        return self.pair_table[h1, h2]                           # [B,T]

    def _step(self, batch):
        X, h1, h2 = batch["input_embeds"], batch["h1"], batch["h2"]
        emis_p, g, c = self(X)
        tags = self._pair_labels(h1, h2)
        crf = _dcrf_nll(emis_p, c, self.nsw_pair, self.stay_bonus, tags)
        loss = crf + self.gate_reg * (1.0 - g).mean()
        return loss, crf, g, c, emis_p, tags

    def training_step(self, batch, _):
        loss, crf, g, c, _, _ = self._step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/crf_loss", crf)
        self.log("train/gate", g.mean())
        return loss

    @torch.no_grad()
    def _accuracy(self, emis_p, c, h1, h2):
        pred = _dcrf_viterbi(emis_p, c, self.nsw_pair, self.stay_bonus)
        pair_true = self.pair_table[h1, h2]
        pair_acc = (pred == pair_true).float().mean()
        # per-haplotype: both stored sorted (pi<=pj), compare to sorted truth
        pred_lo, pred_hi = self.pi[pred], self.pj[pred]
        t_lo = torch.minimum(h1, h2)
        t_hi = torch.maximum(h1, h2)
        hap_acc = ((pred_lo == t_lo).float() + (pred_hi == t_hi).float()).mean() / 2
        return pair_acc, hap_acc

    def validation_step(self, batch, _):
        loss, crf, g, c, emis_p, _ = self._step(batch)
        pair_acc, hap_acc = self._accuracy(emis_p, c, batch["h1"], batch["h2"])
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/pair_acc", pair_acc, prog_bar=True)
        self.log("val/hap_acc", hap_acc, prog_bar=True)
        self.log("val/gate", g.mean())
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        if self.warmup_steps > 0:
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: min(1.0, s / self.warmup_steps))
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5)
        return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val/loss"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--workdir", default="/workdir/esb33")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--limit-n", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gate-reg", type=float, default=0.05)
    p.add_argument("--time-local-emis", action="store_true")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--homo-penalty", type=float, default=0.0,
                   help="Subtract from homozygous pair emissions (het prior); "
                        "counters the all-homozygous collapse of single-read diploid.")
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--run-name", default="diploid-pair")
    return p.parse_args()


def main():
    args = parse_args()
    workdir = Path(args.workdir)
    ckpt_dir = workdir / "checkpoints" / args.run_name
    log_dir = workdir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _ = make_diploid_splits(
        args.data, args.num_parents, args.val_frac, args.test_frac,
        limit_n=args.limit_n)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = GRITSCRFDiploid(
        num_parents=args.num_parents, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, lr=args.lr, gate_reg=args.gate_reg,
        time_local_emis=args.time_local_emis, warmup_steps=args.warmup_steps,
        homo_penalty=args.homo_penalty)

    callbacks = [
        ModelCheckpoint(dirpath=str(ckpt_dir), monitor="val/loss", mode="min",
                        save_top_k=2, filename="d-{epoch:02d}-{val/loss:.3f}"),
        EarlyStopping(monitor="val/loss", mode="min", patience=args.patience),
    ]
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, callbacks=callbacks,
        logger=TensorBoardLogger(str(log_dir), name=args.run_name),
        accelerator="auto", devices=args.devices, precision=args.precision,
        gradient_clip_val=1.0)
    trainer.fit(model, train_loader, val_loader)
    print(f"Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
