#!/usr/bin/env python
"""
Export the raw encoder hidden state H, aligned to REAL genomic bp position
(not window index -- different individuals' windows don't necessarily cover
identical bins, so alignment must go through raw.npy.bins.tsv), for the
first N Mb of chr1, for Oh43 (inbred), Il14H (inbred), and Oh43xIl14H
(hybrid), all under the same checkpoint. Also fits a logistic-regression
"predicted heterozygosity" probe (same design as probe_het_signal.py) on
the already-exported random-window H, then scores this contiguous stretch
with it -- a genome-position track of the probe's output is the intended
"not a PCA" visualization: no dimensionality reduction, just what the
already-validated linear probe reads out of H at each site along the genome.

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src \
      /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/export_encoder_genomic_track.py \
        --max-bp 10000000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src")
sys.path.insert(0, str(Path(__file__).parent))
from python.crf.train_diploid import GRITSCRFDiploid, _founder_affinity  # noqa: E402

GRITS_WORKDIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
SIMVAL_EVAL = GRITS_WORKDIR / "scratch/simval_eval"
ENCODER_EXPORT = GRITS_WORKDIR / "results/encoder_export/diploid-affinity-sim512-h3"
OUT_DIR = GRITS_WORKDIR / "results/encoder_genomic_track"
CKPT = (GRITS_WORKDIR / "checkpoints/diploid-affinity-sim512-h3/"
        "d-epoch=04-val_pair_acc=0.6179.ckpt")
K = 24
ROWS = {
    "Oh43 (inbred)": "IDX-INBRED__Oh43__0.1x",
    "Il14H (inbred)": "IDX-INBRED__Il14H__0.1x",
    "Oh43xIl14H (hybrid)": "IDX-HYB__Oh43xIl14H__0.1x",
}
KIND_HOMO_SCALE = {"IDX-INBRED__Oh43__0.1x": 0.0, "IDX-INBRED__Il14H__0.1x": 0.0,
                    "IDX-HYB__Oh43xIl14H__0.1x": 1.0}


class HCapture:
    def __init__(self):
        self.H = None

    def hook(self, module, inp, out):
        self.H = out.detach()


def fit_probe():
    """Same probe as probe_het_signal.py's kind-level test, refit here so
    this script is self-contained -- reuses the already-exported random
    windows (no re-forward-pass needed for fitting)."""
    H_in1 = np.load(ENCODER_EXPORT / "IDX-INBRED__Oh43__0.1x.npz")["H"]
    H_in2 = np.load(ENCODER_EXPORT / "IDX-INBRED__Il14H__0.1x.npz")["H"]
    H_hyb = np.load(ENCODER_EXPORT / "IDX-HYB__Oh43xIl14H__0.1x.npz")["H"]
    X = np.concatenate([H_in1.reshape(-1, 256), H_in2.reshape(-1, 256), H_hyb.reshape(-1, 256)])
    y = np.concatenate([np.zeros(H_in1.shape[0] * H_in1.shape[1]),
                         np.zeros(H_in2.shape[0] * H_in2.shape[1]),
                         np.ones(H_hyb.shape[0] * H_hyb.shape[1])]).astype(np.int64)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-bp", type=int, default=10_000_000,
                     help="How much of chr1 (from position 0) to export, per individual.")
    ap.add_argument("--contig", default="chr1")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRITSCRFDiploid.load_from_checkpoint(str(CKPT), map_location=device).eval().to(device)
    cap = HCapture()
    model.encoder.pos_encoder.register_forward_hook(cap.hook)

    print("Fitting linear het-probe on already-exported random-window H...")
    clf = fit_probe()
    w, b = clf.coef_[0], clf.intercept_[0]
    dim_order = np.argsort(-np.abs(w))  # most-discriminative dims first, for the heatmap

    for label, row_id in ROWS.items():
        row_dir = SIMVAL_EVAL / row_id
        bins_df = pd.read_csv(row_dir / "raw.npy.bins.tsv", sep="\t")
        full = np.load(row_dir / "windowed_k24_fixdrop23.npy", mmap_mode="r")

        # Which windows fall in [0, max_bp) of the target contig? Window w's
        # bp span = bins_df rows [w*512:(w+1)*512] (matches
        # ropebwt_npy_to_matrix.py's windowing: contiguous window_size row
        # chunks per contig, in on-disk order).
        contig_rows = bins_df[bins_df["contig"] == args.contig]
        n_windows_contig = len(contig_rows) // 512
        win_start_bp = np.array([contig_rows.iloc[w * 512]["bin"] * 256 for w in range(n_windows_contig)])
        sel = np.flatnonzero(win_start_bp < args.max_bp)
        if len(sel) == 0:
            raise ValueError(f"{row_id}: no windows found under {args.max_bp}bp on {args.contig}")
        print(f"{label}: {len(sel)} windows covering up to bp {win_start_bp[sel[-1]] + 512*256:,} on {args.contig}")

        feats_all = np.asarray(full[:, :, :K]).astype(np.float32)
        ext_vec = _founder_affinity(feats_all)
        homo_scale_val = KIND_HOMO_SCALE[row_id]

        bp_positions = []
        Hs = []
        with torch.no_grad():
            for start in range(0, len(sel), 32):
                idx = sel[start:start + 32]
                X = torch.tensor(np.asarray(full[idx, :, :K]).astype(np.float32), device=device)
                B = X.shape[0]
                homo_scale = torch.full((B,), homo_scale_val, device=device)
                ext_t = torch.tensor(ext_vec, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)
                model(X, homo_scale, ext_t)
                Hs.append(cap.H.cpu().numpy())
                for w_idx in idx:
                    rows = contig_rows.iloc[w_idx * 512:(w_idx + 1) * 512]
                    bp_positions.append(rows["bin"].to_numpy() * 256)
        H = np.concatenate(Hs, axis=0)          # [n_sel, 512, 256]
        bp = np.concatenate(bp_positions)        # [n_sel*512]
        Hf = H.reshape(-1, 256)

        probe_score = 1.0 / (1.0 + np.exp(-(Hf @ w + b)))  # sigmoid(logit) = P(het)

        out_path = OUT_DIR / f"{row_id}.npz"
        np.savez_compressed(out_path, H=H, bp=bp, probe_score=probe_score,
                             dim_order=dim_order, probe_weights=w, probe_bias=b,
                             label=label)
        print(f"  wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

    print(f"\nAll exports in {OUT_DIR}")


if __name__ == "__main__":
    main()
