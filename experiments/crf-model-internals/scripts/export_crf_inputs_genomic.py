#!/usr/bin/env python
"""
What actually feeds the CRF decode recursion (_dcrf_viterbi/_dcrf_nll),
EXCLUDING emission scores -- much lower-dimensional and more directly
interpretable than the 256-dim encoder embedding H.

Exactly what the CRF consumes (train_diploid.py):
  emis_p   [B,T,P]  -- EXCLUDED here, per request
  c        [B,T]    -- recombination/switch cost (softplus(recomb_head(...)));
                        transition potential is -c*nsw + stay_bonus*stay_mask.
                        The only per-site, non-emission signal the CRF itself uses.
  nsw_pair [P,P]     -- fixed combinatorial table (# chromosome switches per
                        pair transition), same for every site/individual --
                        nothing to plot as a genomic track.
  stay_bonus         -- single learned SCALAR, same for every site/individual
                        -- reported as a constant, not a track.

Also exports the gate g [B,T] (sigmoid(gate_head(H))) -- not actually passed
to the CRF (it modulates the emission INSIDE the encoder, before emis_p is
formed), but it's the other real per-site "control" signal the model
computes besides raw emission, and directly relevant to "what is the model
doing besides emission."

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src \
      /home/zrm22/mambaforge/envs/phg-ml/bin/python scripts/export_crf_inputs_genomic.py --max-bp 10000000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src")
sys.path.insert(0, str(Path(__file__).parent))
from python.crf.train_diploid import GRITSCRFDiploid, _founder_affinity  # noqa: E402

GRITS_WORKDIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir")
SIMVAL_EVAL = GRITS_WORKDIR / "scratch/simval_eval"
OUT_DIR = GRITS_WORKDIR / "results/crf_inputs_genomic"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-bp", type=int, default=10_000_000)
    ap.add_argument("--contig", default="chr1")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRITSCRFDiploid.load_from_checkpoint(str(CKPT), map_location=device).eval().to(device)
    print(f"stay_bonus (single global learned scalar, same every site/individual): "
          f"{model.stay_bonus.item():.4f}")
    print(f"nsw_pair: fixed [P={model.P},P={model.P}] switch-count table, not a genomic track\n")

    for label, row_id in ROWS.items():
        row_dir = SIMVAL_EVAL / row_id
        bins_df = pd.read_csv(row_dir / "raw.npy.bins.tsv", sep="\t")
        full = np.load(row_dir / "windowed_k24_fixdrop23.npy", mmap_mode="r")
        contig_rows = bins_df[bins_df["contig"] == args.contig]
        n_windows_contig = len(contig_rows) // 512
        win_start_bp = np.array([contig_rows.iloc[w * 512]["bin"] * 256 for w in range(n_windows_contig)])
        sel = np.flatnonzero(win_start_bp < args.max_bp)
        print(f"{label}: {len(sel)} windows up to bp {win_start_bp[sel[-1]] + 512*256:,} on {args.contig}")

        feats_all = np.asarray(full[:, :, :K]).astype(np.float32)
        ext_vec = _founder_affinity(feats_all)
        homo_scale_val = KIND_HOMO_SCALE[row_id]

        bp_positions, gs, cs = [], [], []
        with torch.no_grad():
            for start in range(0, len(sel), 32):
                idx = sel[start:start + 32]
                X = torch.tensor(np.asarray(full[idx, :, :K]).astype(np.float32), device=device)
                B = X.shape[0]
                homo_scale = torch.full((B,), homo_scale_val, device=device)
                ext_t = torch.tensor(ext_vec, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)
                _, g, c = model(X, homo_scale, ext_t)
                gs.append(g.cpu().numpy())
                cs.append(c.cpu().numpy())
                for w_idx in idx:
                    rows = contig_rows.iloc[w_idx * 512:(w_idx + 1) * 512]
                    bp_positions.append(rows["bin"].to_numpy() * 256)
        g_arr = np.concatenate(gs, axis=0).reshape(-1)
        c_arr = np.concatenate(cs, axis=0).reshape(-1)
        bp = np.concatenate(bp_positions)

        out_path = OUT_DIR / f"{row_id}.npz"
        np.savez_compressed(out_path, bp=bp, gate=g_arr, recomb_cost=c_arr,
                             stay_bonus=model.stay_bonus.item(), label=label)
        print(f"  wrote {out_path}  gate mean={g_arr.mean():.4f}  recomb_cost mean={c_arr.mean():.4f}")

    print(f"\nAll exports in {OUT_DIR}")


if __name__ == "__main__":
    main()
