"""
Held-out test-set evaluator for GRITSCRFHaploid checkpoints.

Rebuilds the *identical* deterministic split used in training (make_splits is a
head-slice, no shuffle), then reports founder accuracy on the test split with
both the full Viterbi decode and an emission-only argmax (how much the
transitions add). Works for every arm (full / window-c / no-transition / size
sweep) because the decode path is the model's own `decode()`.

Usage:
    pixi run --environment gpu python src/python/crf/eval_test.py \
        --ckpt <path.ckpt> \
        --data /workdir/.../fullMaizeDataset_all_diploid.npy \
        --limit-n 250000 --split test
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from python.crf.train_haploid import GRITSCRFHaploid, make_splits
from python.crf.metrics import breakpoint_counts, prf

TOLS = (0, 2)


def parse_args():
    p = argparse.ArgumentParser(description="GRITS-CRF held-out evaluator")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--limit-n", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--split", choices=["test", "val"], default="test",
                   help="Which held-out split to score (val for the sanity check).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--workdir", default="/workdir/esb33",
                   help="Results summary is appended under <workdir>/results/.")
    p.add_argument("--tag", default="",
                   help="Optional label for the results line (e.g. arm name).")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, ds, device, batch_size, num_workers):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    n = vit_correct = emis_correct = 0
    bp = {t: {"tp_prec": 0, "n_pred": 0, "tp_rec": 0, "n_true": 0} for t in TOLS}
    for batch in loader:
        X = batch["input_embeds"].to(device)
        tags = batch["labels"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emis_f, g, c = model(X)
        pred_vit = model.decode(emis_f, c)              # respects no_transition
        pred_emis = emis_f.argmax(dim=-1)               # emission-only
        vit_correct += (pred_vit == tags).sum().item()
        emis_correct += (pred_emis == tags).sum().item()
        n += tags.numel()
        for t in TOLS:
            for k, v in breakpoint_counts(pred_vit, tags, tol=t).items():
                bp[t][k] += v
    return vit_correct / n, emis_correct / n, n, bp


def main():
    args = parse_args()
    device = torch.device("cuda")

    model = GRITSCRFHaploid.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)

    data = np.load(args.data, allow_pickle=True, mmap_mode="r")
    train_ds, val_ds, test_ds = make_splits(
        args.data, args.num_parents, args.val_frac, args.test_frac,
        limit_n=args.limit_n)
    ds = test_ds if args.split == "test" else val_ds

    vit_acc, emis_acc, n, bp = evaluate(model, ds, device,
                                        args.batch_size, args.num_workers)

    tag = args.tag or Path(args.ckpt).parent.name
    print(f"\n[{tag}] split={args.split}  N_sites={n:,}")
    print(f"  Viterbi (full decode) acc : {vit_acc:.4f}")
    print(f"  emission-only argmax  acc : {emis_acc:.4f}")
    print(f"  Viterbi - emission        : {vit_acc - emis_acc:+.4f}")
    print(f"  no_transition arm         : {model.no_transition}")
    bp_str = {}
    for t in TOLS:
        p, r, f1 = prf(bp[t])
        bp_str[t] = (p, r, f1)
        print(f"  breakpoint P/R/F1 (±{t})  : {p:.3f} / {r:.3f} / {f1:.3f}  "
              f"(pred={bp[t]['n_pred']:,} true={bp[t]['n_true']:,})")

    res_dir = Path(args.workdir) / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    bp_fields = "\t".join(
        f"bpP{t}={bp_str[t][0]:.3f}\tbpR{t}={bp_str[t][1]:.3f}\t"
        f"bpF{t}={bp_str[t][2]:.3f}" for t in TOLS)
    line = (f"{tag}\tsplit={args.split}\tN={n}\tviterbi={vit_acc:.4f}\t"
            f"emis_only={emis_acc:.4f}\t{bp_fields}\tckpt={args.ckpt}\n")
    with open(res_dir / "maize_eval.tsv", "a") as f:
        f.write(line)
    print(f"  appended → {res_dir / 'maize_eval.tsv'}")


if __name__ == "__main__":
    main()
