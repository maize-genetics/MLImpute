"""
Held-out test-set evaluator for GRITSCRFDiploid checkpoints.

Mirrors eval_test.py but for the diploid pair-state model: loads a
`train_diploid.py` checkpoint, decodes the shared test split with the same
Viterbi used in training, and reports pair_acc and (sorted) hap_acc so the
number is directly comparable to the diploid HMM baseline (eval_diploid_hmm.py)
and the val-time metrics.

Usage:
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
      .pixi/envs/gpu/bin/python src/python/crf/eval_test_diploid.py \
        --ckpt /workdir/esb33/checkpoints/diploid-d128l4/d-epoch=04-....ckpt \
        --data /workdir/esb33/data/training/sim_diploid_512.npy \
        --limit-n 100000 --split test
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from python.crf.train_diploid import (
    GRITSCRFDiploid, make_diploid_splits, _dcrf_viterbi)


def parse_args():
    p = argparse.ArgumentParser(description="GRITS diploid CRF held-out evaluator")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--limit-n", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--workdir", default="/workdir/esb33")
    p.add_argument("--tag", default="")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, ds, device, batch_size, num_workers):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    pair_correct = hap_correct = total = 0
    homo_pred = 0
    for batch in loader:
        X = batch["input_embeds"].to(device)
        h1 = batch["h1"].to(device)
        h2 = batch["h2"].to(device)
        emis_p, _, c = model(X)
        # Reuse the model's own Viterbi + sorted accuracy convention.
        pred = _dcrf_viterbi(emis_p, c, model.nsw_pair, model.stay_bonus)
        pair_true = model.pair_table[h1, h2]
        pair_correct += (pred == pair_true).sum().item()
        pred_lo, pred_hi = model.pi[pred], model.pj[pred]
        t_lo = torch.minimum(h1, h2)
        t_hi = torch.maximum(h1, h2)
        hap_correct += ((pred_lo == t_lo).sum() + (pred_hi == t_hi).sum()).item()
        homo_pred += (pred_lo == pred_hi).sum().item()
        total += pair_true.numel()
    return (pair_correct / total, hap_correct / (2 * total),
            homo_pred / total)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRITSCRFDiploid.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)

    _, val_ds, test_ds = make_diploid_splits(
        args.data, args.num_parents, args.val_frac, args.test_frac,
        limit_n=args.limit_n)
    ds = test_ds if args.split == "test" else val_ds

    pair_acc, hap_acc, homo_pred = evaluate(
        model, ds, device, args.batch_size, args.num_workers)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nckpt={Path(args.ckpt).name}")
    print(f"split={args.split}  params={n_params:,}  "
          f"homo_penalty={model.homo_penalty}")
    print(f"  pair_acc = {pair_acc:.4f}")
    print(f"  hap_acc  = {hap_acc:.4f}")
    print(f"  predicted-homozygous fraction = {homo_pred:.4f}")

    res = Path(args.workdir) / "results" / "diploid_eval.tsv"
    res.parent.mkdir(parents=True, exist_ok=True)
    label = args.tag or Path(args.ckpt).parent.name
    with open(res, "a") as f:
        f.write(f"crf-diploid({label})\tsplit={args.split}\t"
                f"params={n_params}\tpair_acc={pair_acc:.4f}\t"
                f"hap_acc={hap_acc:.4f}\thomo_pred={homo_pred:.4f}\t"
                f"ckpt={Path(args.ckpt).name}\n")
    print(f"appended → {res}")


if __name__ == "__main__":
    main()
