"""
Diagnostic sweep: does a larger recomb_aux_weight actually drive c_t to
anti-correlate with the hidden recomb rate?

Runs N raw training steps (no Lightning) on the smoke file at several weights
and reports the trajectory of the per-window corr(c_t, log_rate). If even a
large weight cannot push corr negative, the aux mechanism is broken; if it can,
we pick a weight for a full run.

Usage:
    pixi run --environment gpu python src/python/crf/sweep_recomb_aux.py \
        --data /workdir/esb33/data/training/sim_coal_e2.npy --steps 400
"""

import argparse
import numpy as np
import torch

from python.crf.train_haploid import (
    GRITSCRFHaploid, PreWindowedHaploidDataset, _window_corr,
)
from torch.utils.data import DataLoader


def run_weight(data, weight, steps, device, lr, warmup):
    torch.manual_seed(0)
    ds = PreWindowedHaploidDataset(data, num_parents=24)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    model = GRITSCRFHaploid(num_parents=24, time_local_emis=True,
                            warmup_steps=warmup,
                            recomb_aux_weight=weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    corrs, c_sds = [], []
    step = 0
    model.train()
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, crf_loss, g, c, _, corr = model._step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            corrs.append(float(corr))
            c_sds.append(float(c.float().std()))
            step += 1
    # report mean over first/last 50 steps
    head = np.mean(corrs[:50])
    tail = np.mean(corrs[-50:])
    c_tail = np.mean(c_sds[-50:])
    return head, tail, c_tail


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weights", default="0,5,50,200,500")
    args = p.parse_args()

    device = torch.device("cuda")
    data = np.load(args.data, mmap_mode="r")
    print(f"data {args.data}  shape={data.shape}")
    K = 24
    assert data.shape[-1] == K + 3, f"need K+3 layout, got {data.shape[-1]}"

    weights = [float(w) for w in args.weights.split(",")]
    print(f"\n{'weight':>8} {'corr_head':>10} {'corr_tail':>10} {'c_sd_tail':>10}")
    print("-" * 42)
    for w in weights:
        head, tail, c_sd = run_weight(data, w, args.steps, device,
                                      args.lr, args.warmup)
        print(f"{w:>8.0f} {head:>10.4f} {tail:>10.4f} {c_sd:>10.4f}")


if __name__ == "__main__":
    main()
