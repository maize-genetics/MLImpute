#!/usr/bin/env python
"""
Resource-checked parallel orchestrator for run_ril2_noerr_sweep.py.

Each pair's full pipeline (build reads -> align both bin-sizes -> score) is
CPU-heavy during refmap (-t 20 threads) and GPU-heavy during inference (one
CRF forward pass per window, one process's whole torch context). Running N
pairs concurrently, each pinned to its own GPU via CUDA_VISIBLE_DEVICES,
gets near-linear speedup up to min(usable GPUs, cores/20, n_pairs) -- this
script measures those three numbers instead of assuming a sequential run is
the only option.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PAIRS = ["B73xOh43", "B73xCML103", "B97xCML103", "Il14HxB97", "Oh43xIl14H"]
SCRIPT_DIR = Path(__file__).parent
WORKER = SCRIPT_DIR / "run_ril2_noerr_sweep.py"
LOG_DIR = Path("results/ril2_error_regions/noerr_0.5x/logs")
REFMAP_THREADS = 20
MIN_FREE_GPU_MB = 8000  # generous margin over the ~5M-param model + one window's activations


def check_resources():
    gpus = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
        for line in out.strip().splitlines():
            idx, free_mb = [x.strip() for x in line.split(",")]
            gpus.append((int(idx), int(free_mb)))
    except Exception as e:
        print(f"WARNING: nvidia-smi check failed ({e}), assuming no GPU available")

    usable_gpus = [idx for idx, free in gpus if free >= MIN_FREE_GPU_MB]

    n_cores = os.cpu_count() or 1
    load1, _, _ = os.getloadavg()
    free_cores = max(1, n_cores - int(round(load1)))
    max_by_cpu = max(1, free_cores // REFMAP_THREADS)

    print("=== resource check ===")
    print(f"GPUs: {gpus} (free MiB) -> usable (>= {MIN_FREE_GPU_MB}MB free): {usable_gpus}")
    print(f"CPU cores: {n_cores} total, load1={load1:.1f}, ~{free_cores} free "
          f"-> up to {max_by_cpu} concurrent refmap (-t {REFMAP_THREADS}) procs")
    return usable_gpus, max_by_cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-workers", type=int, default=None, help="override the auto-detected concurrency")
    cli = ap.parse_args()

    usable_gpus, max_by_cpu = check_resources()
    n_workers = cli.max_workers or max(1, min(len(usable_gpus) or 1, max_by_cpu, len(PAIRS)))
    print(f"-> running {n_workers} pairs concurrently\n")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env_base = os.environ.copy()
    env_base["LD_LIBRARY_PATH"] = ""
    env_base["PYTHONPATH"] = "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src"

    pending = list(PAIRS)
    running = {}  # pair -> (Popen, gpu_id, log_fh)
    t0 = time.time()

    while pending or running:
        while pending and len(running) < n_workers:
            pair = pending.pop(0)
            gpu_id = usable_gpus[len(running) % len(usable_gpus)] if usable_gpus else 0
            env = {**env_base, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            log_fh = open(LOG_DIR / f"{pair}.log", "w")
            print(f"[{time.time()-t0:6.0f}s] launching {pair} on GPU {gpu_id}")
            proc = subprocess.Popen(
                [sys.executable, str(WORKER), "--pair", pair],
                cwd=str(Path.cwd()), env=env, stdout=log_fh, stderr=subprocess.STDOUT)
            running[pair] = (proc, gpu_id, log_fh)

        time.sleep(5)
        for pair, (proc, gpu_id, log_fh) in list(running.items()):
            ret = proc.poll()
            if ret is not None:
                log_fh.close()
                status = "OK" if ret == 0 else f"FAILED (exit {ret})"
                print(f"[{time.time()-t0:6.0f}s] {pair} done on GPU {gpu_id}: {status}")
                del running[pair]

    print(f"\nall pairs done in {time.time()-t0:.0f}s, aggregating...")
    subprocess.run([sys.executable, str(WORKER), "--aggregate-only"],
                    env=env_base, check=True)


if __name__ == "__main__":
    main()
