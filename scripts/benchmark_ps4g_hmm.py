#!/usr/bin/env python
"""Benchmark the PS4G HMM path finder: Kotlin CLI vs Python CLI on CPU vs Python CLI on GPU.

Generates synthetic PS4G files at a few size tiers (via generate_synthetic_ps4g.py),
then times full end-to-end CLI invocations (subprocess wall-clock, matching what a real
user would experience - includes JVM startup / Python+CUDA startup) for both haploid and
diploid path finding, across all three implementations.

Only uses the standard library so it can be run with a plain system `python3` (it shells
out to `pixi run` for the Python CLI to get torch/pandas, and to the built Kotlin binary
directly for the Kotlin CLI).

Usage:
    python scripts/benchmark_ps4g_hmm.py \\
        --grits-root . \\
        --kotlin-bin /path/to/phg_v2_bench/build/install/phg/bin/phg \\
        --work-dir /tmp/ps4g_bench \\
        --output-csv scripts/benchmark_results/ps4g_hmm_benchmark.csv
"""

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

SIZE_TIERS = [
    # (label, positions, gametes, repeats)
    ("small", 1_000, 10, 3),
    ("medium", 10_000, 10, 3),
    ("large", 100_000, 10, 3),
    ("wide", 500, 20, 3),
]

PATH_TYPES = ["haploid", "diploid"]


def generate_ps4g(grits_root: Path, work_dir: Path, label: str, positions: int, gametes: int, seed: int = 1) -> Path:
    out_file = work_dir / f"{label}.ps4g"
    subprocess.run(
        [
            sys.executable,
            str(grits_root / "scripts" / "generate_synthetic_ps4g.py"),
            "--positions", str(positions),
            "--gametes", str(gametes),
            "--seed", str(seed),
            "--output", str(out_file),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return out_file


def time_command(cmd, cwd=None, timeout=600, extra_env=None):
    env = {**os.environ, **extra_env} if extra_env else None
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return elapsed


def kotlin_cmd(kotlin_bin, ps4g_file, out_dir, path_type, inbreed_coef):
    cmd = [
        str(kotlin_bin), "impute-path-from-ps4g",
        "--read-file", str(ps4g_file),
        "--out-path-dir", str(out_dir),
        "--path-type", path_type,
        "--prob-correct", "0.98",
        "--prob-same", "0.9999",
        "--bin-size", "256",
    ]
    if path_type == "diploid":
        cmd += ["--inbreed-coef", str(inbreed_coef)]
    return cmd


def python_cmd(python_bin, ps4g_file, out_dir, path_type, device, inbreed_coef):
    cmd = [
        str(python_bin), "src/python/hmm/impute_ps4g.py",
        "--read-file", str(ps4g_file),
        "--out-path-dir", str(out_dir),
        "--path-type", path_type,
        "--prob-correct", "0.98",
        "--prob-same", "0.9999",
        "--bin-size", "256",
        "--device", device,
    ]
    if path_type == "diploid":
        cmd += ["--inbreed-coef", str(inbreed_coef)]
    return cmd


def check_cuda_available(python_gpu_bin: Path) -> bool:
    result = subprocess.run(
        [str(python_gpu_bin), "-c", "import torch; print(torch.cuda.is_available())"],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "True"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grits-root", required=True, type=Path)
    parser.add_argument("--kotlin-bin", required=True, type=Path)
    parser.add_argument("--python-cpu-bin", required=True, type=Path, help="Python interpreter from a CPU-only torch env.")
    parser.add_argument("--python-gpu-bin", default=None, type=Path, help="Python interpreter from a CUDA torch env.")
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--java-home", default=None, help="JAVA_HOME to use for the Kotlin CLI (needs JDK 21+).")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip python-gpu runs even if CUDA is available.")
    args = parser.parse_args()

    grits_root = args.grits_root.resolve()
    kotlin_bin = args.kotlin_bin.resolve()
    python_cpu_bin = args.python_cpu_bin.resolve()
    python_gpu_bin = args.python_gpu_bin.resolve() if args.python_gpu_bin else None
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    gpu_available = (not args.skip_gpu) and python_gpu_bin is not None and check_cuda_available(python_gpu_bin)
    implementations = ["kotlin", "python-cpu"] + (["python-gpu"] if gpu_available else [])
    print(f"GPU available: {gpu_available}. Implementations: {implementations}")

    rows = []
    for label, positions, gametes, repeats in SIZE_TIERS:
        print(f"\n=== size tier '{label}' ({positions} positions, {gametes} gametes) ===")
        ps4g_file = generate_ps4g(grits_root, work_dir, label, positions, gametes)

        for path_type in PATH_TYPES:
            for impl in implementations:
                times = []
                for rep in range(repeats):
                    out_dir = work_dir / f"{label}_{path_type}_{impl}_{rep}"
                    if out_dir.exists():
                        shutil.rmtree(out_dir)

                    extra_env = None
                    if impl == "kotlin":
                        cmd = kotlin_cmd(kotlin_bin, ps4g_file, out_dir, path_type, 0.0)
                        cwd = None
                        if args.java_home:
                            extra_env = {"JAVA_HOME": args.java_home}
                    elif impl == "python-cpu":
                        cmd = python_cmd(python_cpu_bin, ps4g_file, out_dir, path_type, "cpu", 0.0)
                        cwd = grits_root
                    else:
                        cmd = python_cmd(python_gpu_bin, ps4g_file, out_dir, path_type, "cuda", 0.0)
                        cwd = grits_root

                    try:
                        elapsed = time_command(cmd, cwd=cwd, extra_env=extra_env)
                    except Exception as exc:
                        print(f"  [{path_type}/{impl}] rep {rep} FAILED: {exc}")
                        continue
                    times.append(elapsed)
                    print(f"  [{path_type}/{impl}] rep {rep}: {elapsed:.3f}s")

                if not times:
                    rows.append({
                        "size_label": label, "positions": positions, "gametes": gametes,
                        "path_type": path_type, "implementation": impl,
                        "repeats": 0, "min_s": None, "median_s": None, "mean_s": None,
                    })
                    continue

                rows.append({
                    "size_label": label, "positions": positions, "gametes": gametes,
                    "path_type": path_type, "implementation": impl,
                    "repeats": len(times),
                    "min_s": round(min(times), 4),
                    "median_s": round(statistics.median(times), 4),
                    "mean_s": round(statistics.mean(times), 4),
                })

    with open(args.output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote results to {args.output_csv}")
    print(f"{'size':<8} {'type':<9} {'impl':<11} {'min_s':>10} {'median_s':>10} {'mean_s':>10}")
    for row in rows:
        print(f"{row['size_label']:<8} {row['path_type']:<9} {row['implementation']:<11} "
              f"{str(row['min_s']):>10} {str(row['median_s']):>10} {str(row['mean_s']):>10}")


if __name__ == "__main__":
    main()
