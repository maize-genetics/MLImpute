#!/usr/bin/env python
"""
Phase 4 de-confounder: is B73's elevated hit ratio real sequence relatedness
or reference bias? B73 is both a panel member and the coordinate reference,
so raw elevation over background is expected even with zero bias -- this
script gives an independent, alignment-free relatedness yardstick (mash) to
subtract out.

For each read set's true source assembly (or assemblies, for hybrid/RIL),
regress the observed refmap hit_ratio on mash-derived similarity across the
24 non-B73 founders, then report B73's residual against that fit. A residual
near zero means B73's elevation is explained by relatedness; a large
positive residual is what remains after accounting for it -- the
reference-bias estimate, in the same units as hit_ratio.

Usage:
    refbias_mash.py sketch     # one-time: mash sketch + all-vs-all dist -> distances.tsv
    refbias_mash.py report     # join distances.tsv with refbias results -> residuals
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import refbias_parse as rp  # noqa: E402
import simval_paths as P  # noqa: E402

MASH = Path("/home/zrm22/mambaforge/envs/mash/bin/mash")
INDEX_ASM_DIR = Path("/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2")
HELDOUT_ASM_DIR = P.GRITS_WORKDIR / "data/maize_v2_heldout/asm"
SCRATCH = P.GRITS_WORKDIR / "scratch/refbias_mash"
DIST_TSV = SCRATCH / "distances.tsv"
RESULTS_DIR = P.GRITS_WORKDIR / "results"

HELDOUT_NAMES = rp.HELDOUT_NAMES
ALL_NAMES = rp.PANEL_ORDER + HELDOUT_NAMES  # 25 index + 5 held-out = 30


def assembly_path(name):
    if name in HELDOUT_NAMES:
        return HELDOUT_ASM_DIR / f"{name}.fa"
    return INDEX_ASM_DIR / f"{name}.fa"


def sketch_and_dist():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    msh_path = SCRATCH / "all30.msh"
    fastas = [str(assembly_path(n)) for n in ALL_NAMES]
    for n, fa in zip(ALL_NAMES, fastas):
        if not Path(fa).exists():
            raise SystemExit(f"missing assembly for {n}: {fa}")

    if not msh_path.exists():
        cmd = [str(MASH), "sketch", "-p", "16", "-o", str(msh_path.with_suffix("")), *fastas]
        print("sketching:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print(f"{msh_path} already exists, skipping sketch")

    print("computing all-vs-all mash dist ...")
    proc = subprocess.run([str(MASH), "dist", "-p", "16", str(msh_path), str(msh_path)],
                           capture_output=True, text=True, check=True)
    DIST_TSV.write_text(proc.stdout)
    print(f"wrote {DIST_TSV} ({len(proc.stdout.splitlines())} pairs)")


def load_distance_matrix():
    """{(name_ref, name_query): mash_distance}. mash's ref/query IDs are the
    full FASTA paths sketched into the .msh -- map back to sample names."""
    path_to_name = {str(assembly_path(n)): n for n in ALL_NAMES}
    dist = {}
    with open(DIST_TSV) as f:
        for line in f:
            ref_path, qry_path, d, _p, _shared = line.rstrip("\n").split("\t")
            dist[(path_to_name[ref_path], path_to_name[qry_path])] = float(d)
    return dist


def linreg_fit(x, y):
    """Least-squares y = a*x + b over paired (x,y). Pure-Python, no numpy
    dependency needed for a 24-point fit. Returns (a, b), or None if x has
    no variance (degenerate fit)."""
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var = sum((xi - mean_x) ** 2 for xi in x)
    if var == 0:
        return None
    a = cov / var
    b = mean_y - a * mean_x
    return a, b


def report():
    import json
    dist = load_distance_matrix()
    scratch_root = P.GRITS_WORKDIR / "scratch/refbias"
    results = []
    for d in sorted(scratch_root.iterdir()):
        f = d / "refbias_result.json"
        if f.exists():
            results.append(json.loads(f.read_text()))

    out_path = RESULTS_DIR / "refbias_mash_residual.tsv"
    cols = ["dataset_id", "individual", "arm", "source_assemblies",
            "b73_hit_ratio", "b73_similarity_to_source", "fit_slope", "fit_intercept",
            "b73_predicted_hit_ratio", "b73_residual"]
    n_written = 0
    with open(out_path, "w") as out:
        out.write("\t".join(cols) + "\n")
        for r in results:
            sources = r["parents"] if r["parents"] else (r["individual"],)
            sources = [s for s in sources if s in ALL_NAMES]
            if not sources:
                continue
            # mean similarity (1 - mash distance) from the read set's true
            # source(s) to each of the 24 non-B73 index founders, and to B73.
            def sim_to(founder):
                ds = [dist.get((s, founder), dist.get((founder, s))) for s in sources]
                ds = [d for d in ds if d is not None]
                if not ds:
                    return None
                return 1 - (sum(ds) / len(ds))

            xs, ys = [], []
            for founder in rp.PANEL_ORDER:
                if founder == "B73":
                    continue
                s = sim_to(founder)
                if s is None:
                    continue
                xs.append(s)
                ys.append(r["hit_ratio"][founder])
            b73_sim = sim_to("B73")
            if b73_sim is None or len(xs) < 2:
                continue
            fit = linreg_fit(xs, ys)
            if fit is None:
                continue
            a, b = fit
            pred = a * b73_sim + b
            actual = r["hit_ratio"]["B73"]
            out.write("\t".join(str(x) for x in [
                r["dataset_id"], r["individual"], r["arm"], ",".join(sources),
                f"{actual:.4f}", f"{b73_sim:.4f}", f"{a:.4f}", f"{b:.4f}",
                f"{pred:.4f}", f"{actual - pred:.4f}",
            ]) + "\n")
            n_written += 1
    print(f"wrote {out_path} ({n_written} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sketch", "report"])
    args = ap.parse_args()
    if args.mode == "sketch":
        sketch_and_dist()
    else:
        report()


if __name__ == "__main__":
    main()
