#!/usr/bin/env python
"""
Which encoder dimensions (if any) carry the het/homo signal?

Two complementary tests:
  1. RIL row (Oh43xIl14H), founder-identity-CONTROLLED: real per-site truth
     (via dump_model_internals._window_ril_truth) gives genuinely
     homozygous-truth and heterozygous-truth SITES from the SAME two
     founders in the SAME individual -- isolates zygosity from "which
     founders are present," unlike comparing different inbred lines to a
     different hybrid.
  2. Real inbred (pooled Oh43+Il14H) vs real hybrid, kind-level (confounded
     with founder identity, but the comparison already requested) -- reuses
     the H already exported by export_encoder_output.py.

For each: per-dimension effect size (Cohen's d) ranking, plus a linear
probe (logistic regression) to test whether ANY combination of dimensions
separates het from homo, even if no single dimension does.

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src \
      /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/probe_het_signal.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
from python.crf.train_diploid import GRITSCRFDiploid, _founder_affinity  # noqa: E402
from refbias_parse import split_individual_name  # noqa: E402
from dump_model_internals import _window_ril_truth  # noqa: E402

GRITS_WORKDIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
SIMVAL_EVAL = GRITS_WORKDIR / "scratch/simval_eval"
ENCODER_EXPORT = GRITS_WORKDIR / "results/encoder_export/diploid-affinity-sim512-h3"
CKPT = (GRITS_WORKDIR / "checkpoints/diploid-affinity-sim512-h3/"
        "d-epoch=04-val_pair_acc=0.6179.ckpt")
K = 24


class HCapture:
    def __init__(self):
        self.H = None

    def hook(self, module, inp, out):
        self.H = out.detach()


def export_ril_H_and_truth(n_windows=400, batch_size=64):
    """RIL row: real per-site het/homo truth (h1!=h2) + matching encoder H,
    same footing as export_encoder_output.py but keeping the truth labels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRITSCRFDiploid.load_from_checkpoint(str(CKPT), map_location=device).eval().to(device)
    cap = HCapture()
    model.encoder.pos_encoder.register_forward_hook(cap.hook)

    row_id = "IDX-RIL__Oh43xIl14H__0.1x"
    row_dir = SIMVAL_EVAL / row_id
    gametes = pd.read_csv(row_dir / "raw.npy.gametes.tsv", sep="\t").sort_values("gameteIndex")
    source_names = gametes["sampleName"].tolist()
    dropped_txt = row_dir / "windowed_k24_fixdrop23.dropped_idx.txt"
    dropped_idx = int(dropped_txt.read_text()) if dropped_txt.exists() else 23

    parents = split_individual_name("Oh43xIl14H", "ril")
    h1w_all, h2w_all = _window_ril_truth(row_dir, source_names, dropped_idx,
                                          "IDX-RIL", "Oh43xIl14H", parents)
    n_total = h1w_all.shape[0]
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(n_total, size=min(n_windows, n_total), replace=False))

    full = np.load(row_dir / "windowed_k24_fixdrop23.npy", mmap_mode="r")
    feats_all = np.asarray(full[:, :, :K]).astype(np.float32)
    ext_vec = _founder_affinity(feats_all)

    Hs = []
    with torch.no_grad():
        for start in range(0, len(sel), batch_size):
            idx = sel[start:start + batch_size]
            X = torch.tensor(np.asarray(full[idx, :, :K]).astype(np.float32), device=device)
            B = X.shape[0]
            homo_scale = torch.full((B,), 0.5, device=device)  # RIL production convention
            ext_t = torch.tensor(ext_vec, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)
            model(X, homo_scale, ext_t)
            Hs.append(cap.H.cpu().numpy())
    H = np.concatenate(Hs, axis=0)  # [n_sel, T, 256]
    het = (h1w_all[sel] != h2w_all[sel])  # [n_sel, T] real per-site truth
    return H, het


def cohens_d(a, b):
    """a,b: [N,dims] two groups. Returns per-dim Cohen's d."""
    ma, mb = a.mean(0), b.mean(0)
    sa, sb = a.std(0), b.std(0)
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
    return (ma - mb) / np.maximum(pooled_std, 1e-8)


def probe(X, y, label, top_k=15):
    """Per-dim Cohen's d ranking + logistic-regression linear probe."""
    het_mask = y.astype(bool)
    d = cohens_d(X[het_mask], X[~het_mask])
    order = np.argsort(-np.abs(d))
    print(f"\n=== {label} ===")
    print(f"n_het={het_mask.sum():,}  n_homo={(~het_mask).sum():,}  dims={X.shape[1]}")
    print(f"Top {top_k} dims by |Cohen's d| (het vs homo):")
    for dim in order[:top_k]:
        print(f"  dim {dim:3d}: d={d[dim]:+.3f}  (het mean={X[het_mask,dim].mean():+.3f}, "
              f"homo mean={X[~het_mask,dim].mean():+.3f})")
    print(f"Max |d| = {np.abs(d).max():.3f}   mean |d| across all 256 dims = {np.abs(d).mean():.3f}")
    print("(Cohen's d ~0.2 small, ~0.5 medium, ~0.8 large, by convention)")

    # Linear probe: can ANY combination of dims separate het/homo?
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    acc = clf.score(Xte, yte)
    print(f"Linear probe (logistic regression, held-out): AUC={auc:.4f}  accuracy={acc:.4f}  "
          f"(0.5 AUC = no signal, 1.0 = perfect)")
    top_coef = np.argsort(-np.abs(clf.coef_[0]))[:top_k]
    print(f"Top {top_k} dims by |logistic weight|: {top_coef.tolist()}")
    overlap = len(set(order[:top_k]) & set(top_coef))
    print(f"Overlap between Cohen's-d top-{top_k} and logistic-weight top-{top_k}: {overlap}/{top_k}")
    return d, clf, auc


def main():
    print("Exporting RIL row H + real per-site zygosity truth (founder-identity controlled)...")
    H_ril, het_ril = export_ril_H_and_truth(n_windows=400)
    Xf = H_ril.reshape(-1, H_ril.shape[-1])
    yf = het_ril.reshape(-1).astype(np.int64)
    probe(Xf, yf, "RIL row (Oh43xIl14H), same founders throughout -- true zygosity probe")

    print("\n\nLoading already-exported real inbred vs hybrid H (kind-level, confounded with founder identity)...")
    H_in1 = np.load(ENCODER_EXPORT / "IDX-INBRED__Oh43__0.1x.npz")["H"]
    H_in2 = np.load(ENCODER_EXPORT / "IDX-INBRED__Il14H__0.1x.npz")["H"]
    H_hyb = np.load(ENCODER_EXPORT / "IDX-HYB__Oh43xIl14H__0.1x.npz")["H"]
    X_kind = np.concatenate([H_in1.reshape(-1, 256), H_in2.reshape(-1, 256), H_hyb.reshape(-1, 256)])
    y_kind = np.concatenate([np.zeros(H_in1.shape[0] * H_in1.shape[1]),
                              np.zeros(H_in2.shape[0] * H_in2.shape[1]),
                              np.ones(H_hyb.shape[0] * H_hyb.shape[1])]).astype(np.int64)
    probe(X_kind, y_kind, "Real inbred (pooled) vs real hybrid, kind-level (confounded w/ founder identity)")


if __name__ == "__main__":
    main()
