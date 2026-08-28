#!/usr/bin/env python
"""
Export the RAW encoder hidden state H [n_windows,T,256] for a set of real
rows, all under the SAME checkpoint, for a direct inbred-vs-hybrid
distributional comparison. Encoder-only forward pass (no CRF/Viterbi --
much cheaper than dump_model_internals.py, so a larger window sample is
affordable), GPU if available.

Unlike dump_model_internals.py's H_pca (fit PER-ROW, so components aren't
comparable across rows), this fits ONE PCA basis jointly across all
exported rows, so "PC1" means the same thing for inbred and hybrid --
otherwise a distributional comparison is meaningless.

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src \
      /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/export_encoder_output.py \
        --ckpt checkpoints/diploid-affinity-sim512-h3/d-epoch=04-val_pair_acc=0.6179.ckpt \
        --rows IDX-INBRED__Oh43__0.1x IDX-INBRED__Il14H__0.1x IDX-HYB__Oh43xIl14H__0.1x \
        --n-windows 200
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
from python.crf.train_diploid import GRITSCRFDiploid, _founder_affinity  # noqa: E402
from refbias_parse import split_individual_name  # noqa: E402

GRITS_WORKDIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
SIMVAL_EVAL = GRITS_WORKDIR / "scratch/simval_eval"
OUT_ROOT = GRITS_WORKDIR / "results/encoder_export"
K = 24
KIND_HOMO_SCALE = {"inbred": 0.0, "hybrid": 1.0, "ril": 0.5, "ril2": 0.0}


class HCapture:
    def __init__(self):
        self.H = None

    def hook(self, module, inp, out):
        self.H = out.detach()


def load_model(ckpt_path, device):
    model = GRITSCRFDiploid.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval().to(device)
    cap = HCapture()
    model.encoder.pos_encoder.register_forward_hook(cap.hook)
    return model, cap


def pick_windows(n_total, n_want, seed=0):
    if n_total <= n_want:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_want, replace=False))


@torch.no_grad()
def export_row(model, cap, row_id, n_windows, device, batch_size=64):
    row_dir = SIMVAL_EVAL / row_id
    dataset_class, individual, coverage = row_id.split("__")
    kind = {"INBRED": "inbred", "HYB": "hybrid", "RIL": "ril"}[dataset_class.split("-")[1]]

    gametes = pd.read_csv(row_dir / "raw.npy.gametes.tsv", sep="\t").sort_values("gameteIndex")
    source_names = gametes["sampleName"].tolist()
    dropped_txt = row_dir / "windowed_k24_fixdrop23.dropped_idx.txt"
    dropped_idx = int(dropped_txt.read_text()) if dropped_txt.exists() else 23
    kept_names = [n for i, n in enumerate(source_names) if i != dropped_idx]

    full = np.load(row_dir / "windowed_k24_fixdrop23.npy", mmap_mode="r")
    feats_all = np.asarray(full[:, :, :K]).astype(np.float32)
    ext_vec = _founder_affinity(feats_all)
    n_total = full.shape[0]
    sel = pick_windows(n_total, n_windows)

    homo_scale_val = KIND_HOMO_SCALE[kind]
    Hs = []
    for start in range(0, len(sel), batch_size):
        idx = sel[start:start + batch_size]
        X = torch.tensor(np.asarray(full[idx, :, :K]).astype(np.float32), device=device)
        B = X.shape[0]
        homo_scale = torch.full((B,), homo_scale_val, device=device)
        ext_t = torch.tensor(ext_vec, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)
        model(X, homo_scale, ext_t)
        Hs.append(cap.H.cpu().numpy())
    H = np.concatenate(Hs, axis=0)  # [n_sel, T, 256]

    return dict(row_id=row_id, kind=kind, individual=individual, H=H,
                n_windows_total=n_total, window_idx=sel, founder_names=kept_names,
                ext_emb=ext_vec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--n-windows", type=int, default=200)
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    out_tag = args.out_tag or ckpt_path.parent.name
    out_dir = OUT_ROOT / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cap = load_model(ckpt_path, device)
    print(f"Loaded {ckpt_path} on {device}")

    exported = []
    for row_id in args.rows:
        print(f"Exporting H for {row_id} ({args.n_windows} windows)...")
        res = export_row(model, cap, row_id, args.n_windows, device)
        out_path = out_dir / f"{row_id}.npz"
        np.savez_compressed(out_path, H=res["H"], window_idx=res["window_idx"],
                             ext_emb=res["ext_emb"],
                             founder_names=np.array(res["founder_names"]))
        print(f"  wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)  "
              f"H.shape={res['H'].shape}")
        exported.append(res)

    # Joint PCA across ALL exported rows, so components mean the same thing
    # for every row -- fit on a subsample of sites (SVD on the full stack is
    # unnecessary and slow) then project every site of every row through it.
    print("\nFitting joint PCA across all exported rows...")
    all_H = np.concatenate([r["H"].reshape(-1, r["H"].shape[-1]) for r in exported], axis=0)
    rng = np.random.default_rng(0)
    fit_idx = rng.choice(len(all_H), size=min(200_000, len(all_H)), replace=False)
    mean = all_H[fit_idx].mean(0, keepdims=True)
    Hc = all_H[fit_idx] - mean
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    n_comp = 8
    components = Vt[:n_comp]  # [n_comp, 256]
    explained_var = (S[:n_comp] ** 2) / (len(fit_idx) - 1)
    explained_var_ratio = explained_var / ((S ** 2) / (len(fit_idx) - 1)).sum()

    joint = dict(mean=mean, components=components, explained_var_ratio=explained_var_ratio)
    np.savez_compressed(out_dir / "joint_pca.npz", **joint)
    print(f"Joint PCA explained variance ratio (top 8): {explained_var_ratio}")

    for r in exported:
        Hf = r["H"].reshape(-1, r["H"].shape[-1])
        proj = (Hf - mean) @ components.T
        proj = proj.reshape(r["H"].shape[0], r["H"].shape[1], n_comp)
        np.savez_compressed(out_dir / f"{r['row_id']}.joint_pca_proj.npz",
                             proj=proj, window_idx=r["window_idx"])

    # Quick built-in summary comparison (norms, per-dim mean/std) so there's
    # an immediate read without waiting for a local pull-down.
    print("\n=== Quick summary comparison ===")
    print(f"{'row_id':35s} {'kind':8s} {'H_norm mean':>12s} {'H_norm std':>11s} "
          f"{'per-dim mean(|mu|)':>19s} {'per-dim mean(std)':>18s}")
    for r in exported:
        Hf = r["H"].reshape(-1, r["H"].shape[-1])
        norms = np.linalg.norm(Hf, axis=-1)
        dim_mean = Hf.mean(0)
        dim_std = Hf.std(0)
        print(f"{r['row_id']:35s} {r['kind']:8s} {norms.mean():12.4f} {norms.std():11.4f} "
              f"{np.abs(dim_mean).mean():19.4f} {dim_std.mean():18.4f}")

    print(f"\nWrote encoder exports + joint PCA to {out_dir}")


if __name__ == "__main__":
    main()
