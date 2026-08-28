#!/usr/bin/env python
"""
Tier 0b (see /home/zrm22/.claude/plans/wondrous-discovering-octopus.md):
does decoding across MULTIPLE ADJACENT windows in one Viterbi call (instead
of independent per-512-site-window decodes, which is what production and
last session's dump both do) reduce the real hybrid's spurious switch rate?

Note: last session's model_internals dump sampled windows evenly SPREAD across
the whole chromosome (np.linspace), not adjacent -- concatenating those would
be invalid (fake transitions between genomically-distant windows). This script
instead pulls a genuinely CONTIGUOUS run of windows straight from
windowed_k24_fixdrop23.npy (contiguous by construction --
ropebwt_npy_to_matrix.py windows each contig in on-disk row order) and does a
single fresh (cheap) forward pass, then decodes it two ways.

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src \
      /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/whole_run_decode_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
from python.crf.train_diploid import GRITSCRFDiploid, _founder_affinity, _dcrf_viterbi  # noqa: E402
from refbias_parse import split_individual_name  # noqa: E402

GRITS_WORKDIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
CKPT = (GRITS_WORKDIR / "checkpoints/diploid-affinity-sim512-h3/"
        "d-epoch=04-val_pair_acc=0.6179.ckpt")
ROW = "IDX-HYB__Oh43xIl14H__0.1x"
N_WIN = 24
K = 24


def main():
    row_dir = GRITS_WORKDIR / "scratch/simval_eval" / ROW
    full = np.load(row_dir / "windowed_k24_fixdrop23.npy", mmap_mode="r")
    gametes = pd.read_csv(row_dir / "raw.npy.gametes.tsv", sep="\t").sort_values("gameteIndex")
    source_names = gametes["sampleName"].tolist()
    dropped_txt = row_dir / "windowed_k24_fixdrop23.dropped_idx.txt"
    dropped_idx = int(dropped_txt.read_text()) if dropped_txt.exists() else 23
    kept_names = [n for i, n in enumerate(source_names) if i != dropped_idx]
    name_to_col = {n: i for i, n in enumerate(kept_names)}
    dataset_class, individual, coverage = ROW.split("__")
    parents = split_individual_name(individual, "hybrid")
    h_lo = name_to_col[min(parents)] if False else None  # unordered pair below

    # A genuinely contiguous stretch of windows (bins.tsv row order == chr1 start).
    feats_all = np.asarray(full[:, :, :K]).astype(np.float32)
    ext_vec = _founder_affinity(feats_all)
    X = torch.tensor(feats_all[:N_WIN])            # [N_WIN, 512, 24], contiguous by construction
    B, T, _ = X.shape
    homo_scale = torch.ones(B)
    ext_t = torch.tensor(ext_vec, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)

    model = GRITSCRFDiploid.load_from_checkpoint(str(CKPT), map_location="cpu").eval()
    with torch.no_grad():
        emis_p, g, c = model(X, homo_scale, ext_t)   # [B,T,P], [B,T]

    p_idx = [name_to_col[p] for p in parents]
    true_pair = int(model.pair_table[p_idx[0], p_idx[1]].item())
    nsw_np = model.nsw_pair.numpy()

    # (a) per-window independent decode -- what production/last session do.
    pred_a = _dcrf_viterbi(emis_p, c, model.nsw_pair, model.stay_bonus)   # [B,T]
    acc_a = float((pred_a == true_pair).float().mean())
    sw_a = (nsw_np[pred_a[:, :-1].numpy(), pred_a[:, 1:].numpy()] > 0).sum(-1)  # per window
    print(f"(a) per-window independent decode: {B} windows of T={T}")
    print(f"    pair_acc={acc_a:.4f}  mean switches/window={sw_a.mean():.3f}  "
          f"total switches={sw_a.sum()}")

    # (b) one Viterbi call over the whole concatenated, GENUINELY ADJACENT run.
    emis_whole = emis_p.reshape(1, B * T, -1)
    c_whole = c.reshape(1, B * T)
    pred_b = _dcrf_viterbi(emis_whole, c_whole, model.nsw_pair, model.stay_bonus)  # [1, B*T]
    acc_b = float((pred_b == true_pair).float().mean())
    sw_b_total = int((nsw_np[pred_b[0, :-1].numpy(), pred_b[0, 1:].numpy()] > 0).sum())
    # switches strictly WITHIN a window (excluding the B-1 window-boundary transitions,
    # which (a) never gets to make at all -- an apples-to-apples subset).
    within_mask = np.ones(B * T - 1, dtype=bool)
    within_mask[np.arange(1, B) * T - 1] = False  # boundary transitions
    is_switch_b = nsw_np[pred_b[0, :-1].numpy(), pred_b[0, 1:].numpy()] > 0
    sw_b_within = int(is_switch_b[within_mask].sum())
    print(f"\n(b) single whole-run decode: T={B*T} contiguous sites")
    print(f"    pair_acc={acc_b:.4f}  total switches={sw_b_total} "
          f"(within-window={sw_b_within}, at the {B-1} window boundaries="
          f"{sw_b_total - sw_b_within})")

    print(f"\nDelta: within-window spurious switches {sw_a.sum()} (a) vs {sw_b_within} (b)")


if __name__ == "__main__":
    main()
