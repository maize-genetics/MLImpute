#!/usr/bin/env python
"""
Build the binsize1 (bin-size=1) WITH-sequencing-error tag for the 3 RIL2
pairs that never had one this session (B73xOh43, B97xCML103, Il14HxB97 --
only Oh43xIl14H and B73xCML103 had this already), using the corpus's own
official reads (manifest.tsv). Once built, run_ril2_noerr_sweep.py's
--aggregate-only picks these up automatically (existing_baseline() just
checks for a populated bed/ dir) and fills in the noerr-vs-with-error
SUMMARY table completely.

Single-pair worker mode (--pair) is used by the __main__ launcher below,
which fixes the free-GPU-tracking bug found in run_ril2_noerr_parallel.py:
that script computed each job's GPU from len(running) at launch time
rather than tracking which GPU was actually free, so out-of-order
completion could double-book one GPU while another sat idle. This version
pops/pushes an explicit free-GPU set instead.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import heldout_assembly_eval as hae  # noqa: E402
import simval_eval_one as seo  # noqa: E402
import simval_paths as P  # noqa: E402

WORKTREE_WINDOW_SCRIPT = Path(
    "/local/workdir/zrm22/HackathonJun2026/grits-windowfilter-worktree/"
    "src/python/crf/ropebwt_npy_to_matrix.py")
assert WORKTREE_WINDOW_SCRIPT.exists(), WORKTREE_WINDOW_SCRIPT
hae.nb.WINDOW_SCRIPT = WORKTREE_WINDOW_SCRIPT

MANIFEST = Path("/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/manifest.tsv")
COVERAGE = "0.5"
TAG = "binsize1"
PAIRS = ["B73xOh43", "B97xCML103", "Il14HxB97"]
LOG_DIR = Path("results/ril2_error_regions/noerr_0.5x/logs")
REFMAP_THREADS = 20
MIN_FREE_GPU_MB = 8000


def load_manifest_row(individual, coverage):
    with open(MANIFEST) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["dataset_id"] == "IDX-RIL2" and row["individual"] == individual and row["coverage"] == coverage:
                return row
    raise KeyError(f"no manifest row for IDX-RIL2/{individual}/{coverage}x")


def align_pair(pair):
    row = load_manifest_row(pair, COVERAGE)
    outdir = P.SCRATCH_ROOT / f"IDX-RIL2__{pair}__{COVERAGE}x__{TAG}"
    if (outdir / "bed").exists() and any((outdir / "bed").glob("*.bed")):
        print(f"[{pair}] SKIP, bed/ already populated")
        return
    args = argparse.Namespace(
        sample=f"{pair}_ril2_{TAG}", r1=row["r1_path"], r2=row["r2_path"], outdir=str(outdir),
        arm="refmap", bin_size=1, no_cleanup=False, drop_idx=None,
        max_hit_frac=None, retain_counts=False, kind="ril2", ckpt=str(P.CKPT_DIPLOID),
    )
    info = seo.do_align(args)
    print(f"[{pair}] {info}")


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
        print(f"WARNING: nvidia-smi check failed ({e})")
    usable_gpus = [idx for idx, free in gpus if free >= MIN_FREE_GPU_MB]
    n_cores = os.cpu_count() or 1
    load1, _, _ = os.getloadavg()
    free_cores = max(1, n_cores - int(round(load1)))
    max_by_cpu = max(1, free_cores // REFMAP_THREADS)
    print(f"=== resource check ===\nGPUs free MiB: {gpus} -> usable: {usable_gpus}")
    print(f"CPU cores: {n_cores} total, load1={load1:.1f} -> up to {max_by_cpu} concurrent refmap procs")
    return usable_gpus, max_by_cpu


def main():
    usable_gpus, max_by_cpu = check_resources()
    n_workers = max(1, min(len(usable_gpus) or 1, max_by_cpu, len(PAIRS)))
    print(f"-> running {n_workers} pairs concurrently\n")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env_base = os.environ.copy()
    env_base["LD_LIBRARY_PATH"] = ""
    env_base["PYTHONPATH"] = "/local/workdir/zrm22/HackathonJun2026/test_crf_relatedness/src"

    free_gpus = list(usable_gpus) if usable_gpus else [0]
    pending = list(PAIRS)
    running = {}  # pair -> (Popen, gpu_id, log_fh)
    t0 = time.time()

    while pending or running:
        while pending and len(running) < n_workers and free_gpus:
            pair = pending.pop(0)
            gpu_id = free_gpus.pop(0)
            env = {**env_base, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            log_fh = open(LOG_DIR / f"{pair}_binsize1.log", "w")
            print(f"[{time.time()-t0:6.0f}s] launching {pair} on GPU {gpu_id}")
            proc = subprocess.Popen(
                [sys.executable, __file__, "--pair", pair],
                cwd=str(Path.cwd()), env=env, stdout=log_fh, stderr=subprocess.STDOUT)
            running[pair] = (proc, gpu_id, log_fh)

        time.sleep(5)
        for pair, (proc, gpu_id, log_fh) in list(running.items()):
            ret = proc.poll()
            if ret is not None:
                log_fh.close()
                free_gpus.append(gpu_id)
                status = "OK" if ret == 0 else f"FAILED (exit {ret})"
                print(f"[{time.time()-t0:6.0f}s] {pair} done on GPU {gpu_id}: {status}")
                del running[pair]

    print(f"\nall pairs done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default=None)
    cli = ap.parse_args()
    if cli.pair:
        align_pair(cli.pair)
    else:
        main()
