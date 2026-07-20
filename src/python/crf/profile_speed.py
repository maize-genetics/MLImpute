"""
E8: inference speed/memory profile — where is the decode-time budget, and is a
long-context encoder (Mamba2) actually needed?

Times the two stages of a GRITSCRFHaploid decode separately across window lengths:
the Transformer position-encoder forward (O(T^2) attention) and the Viterbi decode
(the factored stay/switch recursion, O(T*K)). Reports per-stage latency, end-to-end
throughput (sites/s), and peak GPU memory. If the encoder dominates / its memory
blows up super-linearly with T, that is the case for swapping in Mamba2 (E8 lever
a); if Viterbi dominates, optimize the decoder instead.

Usage:
    LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
      .pixi/envs/gpu/bin/python src/python/crf/profile_speed.py \
        --ckpt <ckpt> --window-lengths 256,512,1024,2048,4096 --batch-size 16
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from python.crf.train_haploid import GRITSCRFHaploid


def time_call(fn, reps, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3            # ms/rep


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--window-lengths", default="256,512,1024,2048,4096")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--workdir", default="/workdir/esb33")
    args = p.parse_args()
    device = torch.device("cuda")
    model = GRITSCRFHaploid.load_from_checkpoint(args.ckpt, map_location=device).eval().to(device)
    K = args.num_parents
    B = args.batch_size

    print(f"\n[E8 speed] {Path(args.ckpt).parent.name}  d_model={model.encoder.d_model}  "
          f"batch={B}  device={torch.cuda.get_device_name(0)}")
    hdr = (f"{'T':>6} {'encoder ms':>11} {'viterbi ms':>11} {'total ms':>9} "
           f"{'sites/s':>12} {'peak MB':>9}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for T in [int(x) for x in args.window_lengths.split(",")]:
        X = torch.rand(B, T, K, device=device)
        torch.cuda.reset_peak_memory_stats()

        def fwd():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(X)
        emis_f, g, c = fwd()

        def dec():
            return model.decode(emis_f, c)

        enc_ms = time_call(lambda: fwd(), args.reps)
        dec_ms = time_call(lambda: dec(), args.reps)
        total = enc_ms + dec_ms
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        sps = B * T / (total / 1e3)
        print(f"{T:>6} {enc_ms:>11.2f} {dec_ms:>11.2f} {total:>9.2f} "
              f"{sps:>12,.0f} {peak_mb:>9.0f}")
        rows.append((T, enc_ms, dec_ms, peak_mb))
        del X, emis_f, g, c
        torch.cuda.empty_cache()

    # scaling diagnosis
    if len(rows) >= 2:
        (t0, e0, d0, m0), (t1, e1, d1, m1) = rows[0], rows[-1]
        tr = t1 / t0
        print(f"\n  T x{tr:.0f}: encoder x{e1/max(e0,1e-9):.1f}, viterbi x{d1/max(d0,1e-9):.1f}, "
              f"peak-mem x{m1/max(m0,1e-9):.1f}")
        enc_share = e1 / (e1 + d1) * 100
        print(f"  at T={t1}: encoder is {enc_share:.0f}% of latency → "
              + ("encoder-bound; Mamba2 (E8 lever a) is justified" if enc_share > 60
                 else "decode-bound; optimize Viterbi, Mamba2 not the priority"))

    out = Path(args.workdir) / "results" / "e8_speed.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(f"\n[{Path(args.ckpt).parent.name}] batch={B}\n")
        for T, e, d, m in rows:
            f.write(f"  T={T}: enc {e:.2f}ms viterbi {d:.2f}ms peak {m:.0f}MB\n")
    print(f"\nappended → {out}")


if __name__ == "__main__":
    main()
