"""
E5 hard-cutoff: restrict the decode to an individual's present founders.

Instead of conditioning the encoder (which destabilised bf16 training), apply the
relatedness signal at DECODE time on the already-trained baseline model: estimate
each individual's founder set from the genome-wide affinity (threshold tau on the
per-founder match rate), then set the emissions of absent founders to -inf before
Viterbi. This is stable by construction (no new parameters, no training) and
directly removes the IBD-confusable founders the individual does not carry — the
mechanism behind the read-only -> relatedness ceiling headroom.

Sweeps tau and reports, per breakpoint band: masked-decode accuracy and the
true-founder EXCLUSION rate (fraction of sites whose true founder got masked out —
the risk of too aggressive a cutoff). Compare acc to the no-cutoff baseline and the
relatedness ceiling.

Usage:
    LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
      .pixi/envs/gpu/bin/python src/python/crf/eval_e5_cutoff.py \
        --ckpt <baseline.ckpt> --data /workdir/esb33/data/training/sim_e5_th6.npy \
        --windows-per-individual 100 --taus 0.22,0.24,0.26,0.28
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from python.crf.train_haploid import GRITSCRFHaploid, make_individual_splits

NEG_INF = -1e9
BANDS = [(0, 0, "0 bp"), (1, 2, "1-2 bp"), (3, 5, "3-5 bp"), (6, 10**9, "6+ bp")]


@torch.no_grad()
def collect(model, ds, device, bs):
    """Cache per-window emissions, transition cost, affinity, and truth."""
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    E, C, A, Y = [], [], [], []
    for b in loader:
        X = b["input_embeds"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emis_f, g, c = model(X, None)
        E.append(emis_f.float().cpu())
        C.append(c.float().cpu())
        A.append(b["ext_emb"][:, :, 0])          # raw per-founder affinity [B,K]
        Y.append(b["labels"])
    return torch.cat(E), torch.cat(C), torch.cat(A), torch.cat(Y)


def band_of(switches):
    return next(l for lo, hi, l in BANDS if lo <= switches <= hi)


def score(pred, y, present_full, K):
    """Per-band accuracy + true-founder exclusion rate."""
    stats = {l: [0, 0, 0] for *_, l in BANDS}          # sites, correct, excluded
    stats["all"] = [0, 0, 0]
    for w in range(len(y)):
        sw = int((y[w, 1:] != y[w, :-1]).sum())
        lbl = band_of(sw)
        corr = int((pred[w] == y[w]).sum())
        # true founder excluded where its present flag is False (only real founders)
        tf = y[w].clamp(max=K - 1)
        excl = int((~present_full[w].gather(1, tf[:, None]).squeeze(1)).sum())
        for s in (stats[lbl], stats["all"]):
            s[0] += y.shape[1]; s[1] += corr; s[2] += excl
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--windows-per-individual", type=int, default=100)
    p.add_argument("--taus", default="0.22,0.24,0.26,0.28")
    p.add_argument("--max-windows", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workdir", default="/workdir/esb33")
    args = p.parse_args()
    K = args.num_parents
    Kt = K + 1
    device = torch.device("cuda")
    model = GRITSCRFHaploid.load_from_checkpoint(args.ckpt, map_location=device).eval().to(device)

    _, _, ds = make_individual_splits(args.data, K, 0.10, 0.10, args.windows_per_individual)
    E, C, A, Y = collect(model, ds, device, args.batch_size)
    n = min(args.max_windows, len(Y)) if args.max_windows else len(Y)
    E, C, A, Y = E[:n], C[:n], A[:n], Y[:n]
    nsw = model.nsw.cpu()

    # baseline (no cutoff)
    base_pred = model.crf.viterbi(E.to(device), model._c_eff(C.to(device)), model.nsw).cpu()
    present_all = torch.ones(n, Y.shape[1], Kt, dtype=torch.bool)
    bstats = score(base_pred, Y, present_all, K)

    print(f"\n[E5 hard-cutoff] {Path(args.ckpt).parent.name}  data={Path(args.data).name}  windows={n}")
    print(f"{'tau':>6} {'band':>7} {'acc':>8} {'true-excl%':>11}")
    print("-" * 36)

    def emit(tag, stats):
        for lbl in [l for *_, l in BANDS] + ["all"]:
            s = stats[lbl]
            if s[0]:
                print(f"{tag:>6} {lbl:>7} {s[1]/s[0]:>8.4f} {100*s[2]/s[0]:>10.2f}%")

    emit("none", bstats)
    res = [f"[{Path(args.ckpt).parent.name}] baseline all={bstats['all'][1]/bstats['all'][0]:.4f}"]
    for tau in [float(x) for x in args.taus.split(",")]:
        # present founder per (window, founder); broadcast over sites. Always keep the
        # unknown state (col K). Guarantee >=1 real founder present (keep argmax).
        pres = A >= tau                                      # [n,K]
        keep_top = torch.zeros_like(pres)
        keep_top[torch.arange(n), A.argmax(1)] = True
        pres = pres | keep_top
        pres_full = torch.cat([pres, torch.ones(n, 1, dtype=torch.bool)], dim=1)  # [n,Kt]
        masked = E.clone()
        masked[~pres_full[:, None, :].expand_as(masked)] = NEG_INF
        pred = model.crf.viterbi(masked.to(device), model._c_eff(C.to(device)), model.nsw).cpu()
        pres_sites = pres_full[:, None, :].expand(n, Y.shape[1], Kt)
        st = score(pred, Y, pres_sites, K)
        emit(f"{tau:.2f}", st)
        res.append(f"tau={tau:.2f} all={st['all'][1]/st['all'][0]:.4f} "
                   f"excl={100*st['all'][2]/st['all'][0]:.2f}%")

    out = Path(args.workdir) / "results" / "e5_cutoff.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write("\n" + " | ".join(res) + "\n")
    print(f"\nappended → {out}")


if __name__ == "__main__":
    main()
