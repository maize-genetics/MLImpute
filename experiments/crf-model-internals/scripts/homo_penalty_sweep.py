#!/usr/bin/env python
"""
Tier 0a (see /home/zrm22/.claude/plans/wondrous-discovering-octopus.md):
free, decode-time-only homo_penalty sensitivity sweep, reusing the per-founder
emissions already captured in results/model_internals/*.npz last session --
no simulation, no training, no GPU forward pass (only `stay_bonus` needs the
checkpoint, a scalar).

Recomputes emis_p = emis_f[pi]+emis_f[pj] - pen*homo_mask at several pen
values (the deployed checkpoint uses a fixed pen=3.0 always, or 0 when
homo_scale=0/None) and re-decodes with _dcrf_viterbi, to see whether an
intermediate fixed penalty does better on the real hybrid without retraining.

Restricted to the 24 real founders (P=300 pairs, no null state) since the
dumped emis_f already dropped the null column -- a clean approximation for
this decode-time-only check (the null state is never the right answer on
real reads with actual support).

Usage:
    /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/homo_penalty_sweep.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src")
from python.crf.train_diploid import build_pair_tables, _dcrf_viterbi, GRITSCRFDiploid  # noqa: E402

GRITS_WORKDIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
CKPT = (GRITS_WORKDIR / "checkpoints/diploid-affinity-sim512-h3/"
        "d-epoch=04-val_pair_acc=0.6179.ckpt")
NPZ_DIR = GRITS_WORKDIR / "results/model_internals"
PENALTIES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def sweep_row(row_id, K=24):
    d = np.load(NPZ_DIR / f"{row_id}.npz")
    emis_f = torch.tensor(d["emis_f"], dtype=torch.float32)   # [B,T,24]
    c = torch.tensor(d["recomb_cost"], dtype=torch.float32)   # [B,T]
    true_lo = torch.tensor(d["true_lo"], dtype=torch.long)
    true_hi = torch.tensor(d["true_hi"], dtype=torch.long)

    pi, pj, pair_table, nsw = build_pair_tables(K)
    homo_mask = (pi == pj).float()
    stay_bonus = torch.tensor(float(_stay_bonus()))

    emis_p_raw = emis_f[..., pi] + emis_f[..., pj]             # [B,T,P]
    true_pair = pair_table[true_lo, true_hi]                   # [B,T]

    print(f"\n=== {row_id} ===")
    print(f"{'pen':>5} {'pair_acc':>9} {'hap_acc':>8} {'decoded_switch/win':>19} "
          f"{'het_frac_decoded':>17}")
    rows = []
    for pen in PENALTIES:
        emis_p = emis_p_raw - pen * homo_mask
        pred = _dcrf_viterbi(emis_p, c, nsw, stay_bonus)
        pred_lo, pred_hi = pi[pred], pj[pred]
        pair_acc = (pred == true_pair).float().mean().item()
        hap_acc = (((pred_lo == true_lo).float() + (pred_hi == true_hi).float()) / 2).mean().item()
        nsw_np = nsw.numpy()
        dec_sw = (nsw_np[pred[:, :-1].numpy(), pred[:, 1:].numpy()] > 0).sum(-1).mean()
        het_frac = (homo_mask[pred] < 0.5).float().mean().item()
        print(f"{pen:5.1f} {pair_acc:9.4f} {hap_acc:8.4f} {dec_sw:19.3f} {het_frac:17.4f}")
        rows.append((pen, pair_acc, hap_acc, dec_sw, het_frac))
    return rows


def _stay_bonus():
    # Only need the scalar stay_bonus parameter -- load the checkpoint's state
    # dict directly rather than a full model instantiation+forward pass.
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    return ckpt["state_dict"]["stay_bonus"].item()


def main():
    for row_id in ["IDX-HYB__Oh43xIl14H__0.1x", "IDX-INBRED__Oh43__0.1x",
                    "IDX-INBRED__Il14H__0.1x"]:
        sweep_row(row_id)
    print("\nProduction uses pen=3.0 for hybrid (homo_scale=1.0*3.0), "
          "pen=0.0 for inbred (homo_scale=0.0*3.0) -- both endpoints of this sweep.")


if __name__ == "__main__":
    main()
