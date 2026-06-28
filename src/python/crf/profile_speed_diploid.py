"""
Diploid inference speed/memory profile — where is the decode-time budget at
genome scale, and what should we optimise?

Times the GRITSCRFDiploid pipeline stages across window lengths: the encoder
forward (cell-MLP embed + founder-pool attention + O(T^2) Transformer + emission),
and the P=325 pair-state decode (Viterbi vs forward-backward marginal). Also
micro-benches the cell-MLP alone (the target of the binary-input lookup fast-path:
for a 0/1 matrix it has only two distinct outputs) and measures whole-chromosome
decode throughput at L=100k. Reports per-stage latency, positions/s, peak GPU mem.

Usage:
    LD_LIBRARY_PATH=.pixi/envs/gpu/lib PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
      .pixi/envs/gpu/bin/python src/python/crf/profile_speed_diploid.py \
        --ckpt <diploid.ckpt> --window-lengths 256,512,1024,2048,4096 --batch-size 16
"""

import argparse
import time
from pathlib import Path

import torch

from python.crf.train_diploid import GRITSCRFDiploid, _dcrf_viterbi, _dcrf_marginal


def time_call(fn, reps, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3                # ms/rep


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num-parents", type=int, default=24)
    p.add_argument("--window-lengths", default="256,512,1024,2048,4096")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--chrom-len", type=int, default=100_000,
                   help="length for the whole-chromosome decode throughput test")
    p.add_argument("--workdir", default="/workdir/esb33")
    args = p.parse_args()
    device = torch.device("cuda")
    model = GRITSCRFDiploid.load_from_checkpoint(args.ckpt, map_location=device).eval().to(device)
    K, B, enc = args.num_parents, args.batch_size, model.encoder

    print(f"\n[diploid speed] {Path(args.ckpt).parent.parent.name}  "
          f"d_model={enc.d_model}  P={model.P}  batch={B}  {torch.cuda.get_device_name(0)}")
    hdr = (f"{'T':>6} {'enc ms':>8} {'cell ms':>8} {'cell%':>6} {'viterbi':>8} "
           f"{'margin':>8} {'pos/s(V)':>11} {'peak MB':>9}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for T in [int(x) for x in args.window_lengths.split(",")]:
        X = (torch.rand(B, T, K, device=device) < 0.3).float()    # binary-ish match matrix
        torch.cuda.reset_peak_memory_stats()

        def fwd():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(X)
        emis_p, g, c = fwd()

        def cell_only():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return enc.cell(torch.log1p(X).unsqueeze(-1))      # [B,T,K,d]

        enc_ms = time_call(fwd, args.reps)
        cell_ms = time_call(cell_only, args.reps)
        vit_ms = time_call(lambda: _dcrf_viterbi(emis_p, c, model.nsw_pair, model.stay_bonus),
                           args.reps)
        mar_ms = time_call(lambda: _dcrf_marginal(emis_p, c, model.nsw_pair, model.stay_bonus),
                           max(2, args.reps // 2))
        peak = torch.cuda.max_memory_allocated() / 1e6
        pos_s = B * T / ((enc_ms + vit_ms) / 1e3)
        cell_pct = 100 * cell_ms / max(enc_ms, 1e-9)
        print(f"{T:>6} {enc_ms:>8.2f} {cell_ms:>8.2f} {cell_pct:>5.0f}% {vit_ms:>8.2f} "
              f"{mar_ms:>8.2f} {pos_s:>11,.0f} {peak:>9.0f}")
        rows.append((T, enc_ms, cell_ms, vit_ms, mar_ms, peak))
        del X, emis_p, g, c
        torch.cuda.empty_cache()

    # whole-chromosome single decode throughput (encoder is tiled at 1024)
    L = args.chrom_len
    emis_full = torch.randn(1, L, model.P, device=device)
    c_full = torch.rand(1, L, device=device)
    torch.cuda.reset_peak_memory_stats()
    vit_full = time_call(lambda: _dcrf_viterbi(emis_full, c_full, model.nsw_pair, model.stay_bonus),
                         3)
    peak_full = torch.cuda.max_memory_allocated() / 1e6
    print(f"\n  whole-chrom decode L={L:,}: viterbi {vit_full:.1f} ms "
          f"({L/(vit_full/1e3):,.0f} pos/s) peak {peak_full:.0f} MB")

    if len(rows) >= 2:
        (t0, e0, _, d0, *_), (t1, e1, _, d1, *_) = rows[0], rows[-1]
        enc_share = e1 / (e1 + d1) * 100
        print(f"  T {t0}->{t1}: encoder x{e1/max(e0,1e-9):.1f}, viterbi x{d1/max(d0,1e-9):.1f}; "
              f"at T={t1} encoder is {enc_share:.0f}% of latency")
        print(f"  cell-MLP is ~{rows[2][2]/max(rows[2][1],1e-9)*100:.0f}% of the encoder at "
              f"T={rows[2][0]} → binary-lookup fast-path target")

    out = Path(args.workdir) / "results" / "wg_speed.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(f"\n[{Path(args.ckpt).parent.parent.name} d{enc.d_model}] batch={B}\n")
        for T, e, cm, v, m, pk in rows:
            f.write(f"  T={T}: enc {e:.2f} cell {cm:.2f} viterbi {v:.2f} marginal {m:.2f} "
                    f"peak {pk:.0f}MB\n")
        f.write(f"  whole-chrom L={L}: viterbi {vit_full:.1f}ms peak {peak_full:.0f}MB\n")
    print(f"\nappended → {out}")


if __name__ == "__main__":
    main()
