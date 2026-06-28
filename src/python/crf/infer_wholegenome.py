"""
Whole-chromosome diploid inference — edge-free decoding at genome scale.

The encoder is O(T^2) and capped at ~1024-site context, but the pair-state CRF
decode is LINEAR in length, so it can span a whole chromosome. Tiling a 100k-site
chromosome into independent 1024 windows leaves a decode discontinuity at every
boundary (the state at a tile edge has no neighbouring context). This module fixes
that by running the encoder on OVERLAPPED windows, stitching the per-site emissions
into a full-chromosome track, and running ONE CRF decode over the entire chromosome.

Three modes (set `--mode`):
  tiled         independent non-overlap 1024 windows, decoded separately (baseline;
                edge errors at each boundary).
  overlap-stitch overlapped encoder windows, decoded separately, paths center-cropped
                and stitched (encoder overlap, but still per-window decode).
  whole-chrom   overlapped encoder emissions stitched into one [L,P] track, decoded
                ONCE per chromosome (the edge-free target).

Reuses GRITSCRFDiploid.forward, _dcrf_viterbi/_dcrf_marginal, build_pair_tables,
_founder_affinity (affinity model). Per-chromosome decode (no inter-chromosome
transition). Reports per-class pair/hap and boundary-site accuracy (the edge metric).

Usage:
    LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
      .pixi/envs/gpu/bin/python src/python/crf/infer_wholegenome.py \
        --ckpt <affinity.ckpt> --glob '/workdir/esb33/data/held-out/wg_*_sparse.npy' \
        --mode whole-chrom --decode viterbi --stride 512 --founder-affinity
"""

import argparse
import glob as globmod
from pathlib import Path

import numpy as np
import torch

from python.crf.train_diploid import (GRITSCRFDiploid, _dcrf_viterbi,
                                       _dcrf_marginal, _dcrf_viterbi_factored,
                                       _founder_affinity)

DECODERS = {"viterbi": _dcrf_viterbi, "marginal": _dcrf_marginal}


def _t(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    import time
    return time.perf_counter()


def _starts(L, win, stride):
    s = list(range(0, max(1, L - win + 1), stride))
    if s[-1] != L - win:
        s.append(L - win)
    return s


@torch.no_grad()
def _encode(model, feats, starts, win, ext_emb, device, bs):
    """feats [L,K] -> per-window emis_p [nwin,win,P], c [nwin,win]."""
    Xs = torch.stack([feats[s:s + win] for s in starts])          # [nwin,win,K]
    em, cc = [], []
    for i in range(0, len(starts), bs):
        Xb = Xs[i:i + bs].to(device)
        eb = ext_emb.expand(Xb.shape[0], -1, -1) if ext_emb is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emis_p, _g, c = model(Xb, None, eb)
        em.append(emis_p.float().cpu())
        cc.append(c.float().cpu())
    return torch.cat(em), torch.cat(cc)


def _ownership(starts, L, win, stride):
    """Center-crop boundaries: window i owns output [b_prev, b_i)."""
    half = (stride + win) // 2
    bounds, b_prev = [], 0
    for i, s in enumerate(starts):
        b = L if i == len(starts) - 1 else s + half
        bounds.append((b_prev, b, s))                              # out lo, out hi, win start
        b_prev = b
    return bounds


def affinity_keep(rate, min_gap=0.03, margin=0.02):
    """Sorted-rate largest-gap split: founders above the carried/background step.
    Returns kept founder indices (np). No clear step (gap<min_gap) -> keep all."""
    order = np.argsort(rate)[::-1]
    sr = rate[order]
    gaps = sr[:-1] - sr[1:]
    cut = int(np.argmax(gaps))
    if gaps[cut] < min_gap:                       # flat -> all founders carried
        return np.arange(len(rate))
    return np.where(rate >= sr[cut + 1] + margin)[0]


def decode_chrom(model, feats, ext_emb, mode, decode, win, stride, device, bs,
                 sel_pairs=None):
    """feats [L,K] tensor -> pred pair path [L] (long, cpu). If sel_pairs given,
    decode over that reduced pair-state subset and remap to original indices."""
    L = feats.shape[0]
    nsw, stay = model.nsw_pair, model.stay_bonus
    if sel_pairs is not None:
        nsw = nsw[sel_pairs][:, sel_pairs]         # reduced P'xP' switch matrix
        decode = "viterbi" if decode == "viterbi-factored" else decode
    if decode == "viterbi-factored":            # O(P+K), faster on CPU (exact)
        decoder = lambda e, cc, n, s: _dcrf_viterbi_factored(e, cc, n, s, model.pi, model.pj)
    else:
        decoder = DECODERS[decode]
    eff_stride = win if mode == "tiled" else stride
    starts = _starts(L, win, eff_stride)
    emis_w, c_w = _encode(model, feats, starts, win, ext_emb, device, bs)   # [nwin,win,*]
    bounds = _ownership(starts, L, win, eff_stride)

    def _decode(emis_b, c_b):                       # [1,len,P] -> [len] orig-pair idx
        e = emis_b[..., sel_pairs] if sel_pairs is not None else emis_b
        t0 = _t(device)
        p = decoder(e.to(device), c_b.to(device), nsw, stay)[0].cpu()
        dt = _t(device) - t0
        if sel_pairs is not None:
            p = sel_pairs.cpu()[p]                  # remap reduced -> original pair idx
        return p, dt

    if mode == "whole-chrom":
        P = emis_w.shape[-1]
        emis_full = torch.empty(L, P)
        c_full = torch.empty(L)
        for (lo, hi, s), ew, cw in zip(bounds, emis_w, c_w):
            emis_full[lo:hi] = ew[lo - s:hi - s]
            c_full[lo:hi] = cw[lo - s:hi - s]
        pred, dt = _decode(emis_full.unsqueeze(0), c_full.unsqueeze(0))
        return pred, dt

    # tiled / overlap-stitch: decode each window, center-crop stitch the PATHS
    pred = torch.empty(L, dtype=torch.long)
    dt = 0.0
    for (lo, hi, s), ew, cw in zip(bounds, emis_w, c_w):
        p, d = _decode(ew.unsqueeze(0), cw.unsqueeze(0)); dt += d
        pred[lo:hi] = p[lo - s:hi - s]
    return pred, dt


def acc(model, pred, h1, h2):
    """pred [L] pair idx; h1,h2 [L]. Returns (pair_correct[L] bool, hap_half[L])."""
    pair_true = model.pair_table.cpu()[h1, h2]
    pc = (pred == pair_true)
    lo, hi = model.pi.cpu()[pred], model.pj.cpu()[pred]
    tlo, thi = torch.minimum(h1, h2), torch.maximum(h1, h2)
    hh = ((lo == tlo).float() + (hi == thi).float()) / 2
    return pc, hh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--glob", required=True, help="WG individual .npy files")
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--mode", choices=["tiled", "overlap-stitch", "whole-chrom"],
                   default="whole-chrom")
    p.add_argument("--decode", choices=["viterbi", "marginal", "viterbi-factored"],
                   default="viterbi")
    p.add_argument("--win", type=int, default=1024)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--founder-affinity", action="store_true")
    p.add_argument("--prune-affinity", action="store_true",
                   help="affinity largest-gap select carried founders -> decode the "
                        "reduced pair-state subset (the smaller-CRF speed lever)")
    p.add_argument("--prune-min-gap", type=float, default=0.03)
    p.add_argument("--prune-margin", type=float, default=0.02)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--boundary-d", type=int, default=16,
                   help="report accuracy within +/-d of tile (win) boundaries")
    p.add_argument("--workdir", default="/workdir/esb33")
    args = p.parse_args()
    device = torch.device("cuda")
    model = GRITSCRFDiploid.load_from_checkpoint(args.ckpt, map_location=device).eval().to(device)
    K = args.num_parents

    files = sorted(globmod.glob(args.glob))
    if not files:
        raise SystemExit(f"no files match {args.glob}")
    print(f"\n[whole-genome infer] mode={args.mode} decode={args.decode} "
          f"win={args.win} stride={args.stride} affinity={args.founder_affinity}")
    hdr = (f"{'individual':32s} {'pair':>7} {'hap':>7} {'bnd_pair':>9} {'pos':>10} "
           f"{'P_red':>6} {'dec_ms':>7}")
    print(hdr); print("-" * len(hdr))

    pi_c, pj_c = model.pi.cpu().numpy(), model.pj.cpu().numpy()
    tot_pc = tot_hh = tot_n = 0.0
    tot_bpc = tot_bn = 0.0
    tot_dt = 0.0; tot_pred = 0; nchrom = 0
    rows = []
    for f in files:
        arr = np.load(f)                                          # [NC, L, K+2]
        feats_all = torch.tensor(arr[:, :, :K], dtype=torch.float32)
        h1_all = torch.tensor(np.clip(arr[:, :, K], 0, K).astype(np.int64))
        h2_all = torch.tensor(np.clip(arr[:, :, K + 1], 0, K).astype(np.int64))
        ext = None
        sel_pairs = None
        if args.founder_affinity or args.prune_affinity:
            aff = _founder_affinity(arr[:, :, :K].astype(np.float32))   # [K,2] over whole indiv
            ext = torch.tensor(aff, dtype=torch.float32, device=device).unsqueeze(0)
        if args.prune_affinity:
            sel = affinity_keep(aff[:, 0], args.prune_min_gap, args.prune_margin)
            keep = set(int(i) for i in sel) | {K}    # always retain the null founder
            sp = [pp for pp in range(model.P) if pi_c[pp] in keep and pj_c[pp] in keep]
            sel_pairs = torch.tensor(sp, dtype=torch.long)
        P_red = len(sel_pairs) if sel_pairs is not None else model.P

        ipc = ihh = inn = 0.0
        ibpc = ibn = 0.0; idt = 0.0
        for ch in range(feats_all.shape[0]):
            feats = feats_all[ch]
            L = feats.shape[0]
            pred, dt = decode_chrom(model, feats, ext, args.mode, args.decode,
                                    args.win, args.stride, device, args.batch_size,
                                    sel_pairs=sel_pairs)
            pc, hh = acc(model, pred, h1_all[ch], h2_all[ch])
            ipc += pc.sum().item(); ihh += hh.sum().item(); inn += L; idt += dt
            # boundary mask: within +/-d of any win-multiple (tiled cut points)
            t = torch.arange(L)
            bnd = ((t % args.win) <= args.boundary_d) | ((t % args.win) >= args.win - args.boundary_d)
            ibpc += pc[bnd].sum().item(); ibn += int(bnd.sum())
        name = Path(f).stem
        dec_ms = idt / feats_all.shape[0] * 1e3
        print(f"{name:32s} {ipc/inn:>7.4f} {ihh/inn:>7.4f} {ibpc/ibn:>9.4f} {int(inn):>10,} "
              f"{P_red:>6d} {dec_ms:>7.1f}")
        rows.append(f"{name}={ipc/inn:.4f}/{ihh/inn:.4f}")
        tot_pc += ipc; tot_hh += ihh; tot_n += inn
        tot_bpc += ibpc; tot_bn += ibn
        tot_dt += idt; nchrom += feats_all.shape[0]

    print("-" * len(hdr))
    print(f"{'ALL':32s} {tot_pc/tot_n:>7.4f} {tot_hh/tot_n:>7.4f} {tot_bpc/tot_bn:>9.4f} "
          f"{int(tot_n):>10,} {'':>6} {tot_dt/nchrom*1e3:>7.1f}")
    out = Path(args.workdir) / "results" / "wg_infer.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    prune = "prune " if args.prune_affinity else ""
    with open(out, "a") as fh:
        fh.write(f"\n[{Path(args.ckpt).parent.parent.name} mode={args.mode} "
                 f"decode={args.decode} {prune}stride={args.stride}] {Path(args.glob).name}: "
                 + " | ".join(rows)
                 + f" | ALL={tot_pc/tot_n:.4f}/{tot_hh/tot_n:.4f} "
                 f"bnd_pair={tot_bpc/tot_bn:.4f} dec_ms/chrom={tot_dt/nchrom*1e3:.1f}\n")
    print(f"\nappended → {out}")


if __name__ == "__main__":
    main()
