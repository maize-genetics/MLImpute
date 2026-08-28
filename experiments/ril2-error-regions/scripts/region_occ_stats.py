#!/usr/bin/env python
"""
Item 1 of the RIL2 error-region diagnostics
(/home/zrm22/.claude/plans/wondrous-discovering-octopus.md): "variance of
SSA in error regions" -- clarified by the user as the pangenome-wide
occurrence count that ropebwt3's sampled suffix array (.fmd.ssa) backs.

The .ssa file itself holds no counts (it is a rank->genome-position lookup
table for locate(), see the plan's "Can the .ssa file answer item 1 on its
own?" section) -- occurrence counts are FM-index interval sizes, produced
by a backward search over the .fmd. This script gets them WITHOUT running
refmap or aligning any reads: it tiles B73 reference sequence across each
region and queries `ropebwt3 mem`, a pure string-lookup subcommand that
prints `name start end interval_size` for arbitrary input sequence and
needs only the .fmd (not the .ssa -- that's loaded only with `mem -p`,
which this script does not use).

Benchmarked: index load is ~1.35s regardless of query count; 1000 tiles
processed in ~0.02s beyond that. ~484k tiles (1210 regions x 400 cap)
finishes in well under a minute, not the "minutes" estimated in the plan.
"""
import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

RB3 = "/workdir/zrm22/HackathonJun2026/ropebwt_refMap/ropebwt3-phg/.claude/worktrees/refmap-ps4g-numpy/ropebwt3"
IDX = "/workdir/zrm22/HackathonJun2026/ropebwt_refMap/rope_bwt_index_v2/maizeFastaIndex_SampleContig_v2.fmd"
FASTA = "/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/B73.fa"
FAI = FASTA + ".fai"

TILE_LEN = 150       # matches the simulated read length used throughout this corpus
STRIDE = 500
MAX_TILES = 400       # bounds cost regardless of the 1.7Mb max region width


def load_chrom_lengths():
    lengths = {}
    with open(FAI) as f:
        for line in f:
            name, length = line.split("\t")[:2]
            lengths[name] = int(length)
    return lengths


def tile_positions(chrom_len, start, end):
    """Evenly spaced TILE_LEN windows covering [start,end), capped at MAX_TILES.
    Regions narrower than TILE_LEN get one window centered on the midpoint,
    clipped to chrom bounds (affects 8/210 error, 41/1000 background regions)."""
    width = end - start
    if width < TILE_LEN:
        mid = (start + end) // 2
        s = max(0, min(chrom_len - TILE_LEN, mid - TILE_LEN // 2))
        return [s]
    n_tiles = min(MAX_TILES, max(1, (width - TILE_LEN) // STRIDE + 1))
    last_start = end - TILE_LEN
    if n_tiles == 1:
        return [start]
    step = (last_start - start) / (n_tiles - 1)
    seen, out = set(), []
    for i in range(n_tiles):
        s = start + round(i * step)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def region_key(chrom, s, e):
    return f"{chrom}:{s}-{e}"  # 0-based half-open, internal key (not samtools coords)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--error-json", default="results/ril2_error_regions/error_regions_full.json")
    ap.add_argument("--background-json", default="results/ril2_error_regions/background_regions_full.json")
    ap.add_argument("--out-dir", default="results/ril2_error_regions")
    ap.add_argument("--pilot", type=int, default=0,
                     help="if >0, only process this many regions per group (timing/sanity pilot)")
    cli = ap.parse_args()
    out_dir = Path(cli.out_dir)

    error_regions = json.loads(Path(cli.error_json).read_text())
    bg_regions = json.loads(Path(cli.background_json).read_text())
    if cli.pilot:
        error_regions = error_regions[: cli.pilot]
        bg_regions = bg_regions[: cli.pilot]
    all_regions = [(r, "error") for r in error_regions] + [(r, "background") for r in bg_regions]

    lengths = load_chrom_lengths()

    # tile_key ("chrom:0based_s-0based_e", TILE_LEN wide) -> list of (group_idx) owning it.
    # Dedup by coordinate: two regions tiling to the identical genomic window is a cache
    # hit, not a bug -- both legitimately get the same occurrence count.
    tile_owners = {}
    region_tile_keys = []  # parallel to all_regions
    for r, _kind in all_regions:
        chrom_len = lengths[r["chrom"]]
        positions = tile_positions(chrom_len, r["start"], r["end"])
        keys = []
        for s in positions:
            e = s + TILE_LEN
            key = region_key(r["chrom"], s, e)
            tile_owners.setdefault(key, []).append(len(region_tile_keys))
            keys.append(key)
        region_tile_keys.append(keys)

    print(f"{len(all_regions)} regions ({len(error_regions)} error, {len(bg_regions)} background) "
          f"-> {len(tile_owners)} distinct {TILE_LEN}bp tiles")

    # samtools faidx region syntax is 1-based inclusive.
    regions_txt = out_dir / "occ_tiles.regions.txt"
    with open(regions_txt, "w") as f:
        for key in tile_owners:
            chrom, rest = key.split(":")
            s, e = (int(x) for x in rest.split("-"))
            f.write(f"{chrom}:{s + 1}-{e}\n")

    tiles_fa = out_dir / "occ_tiles.fa"
    with open(tiles_fa, "w") as out_f:
        subprocess.run(["samtools", "faidx", "-r", str(regions_txt), FASTA], stdout=out_f, check=True)

    # N-containing tiles (assembly gaps) must be excluded before aggregation: a poly-N
    # query matches every N-run in every founder in the index, producing an FM-index
    # interval in the hundreds of millions -- a real search result, but semantically
    # meaningless "repetitiveness" (found via a background region that landed entirely
    # inside an N-gap: occ_mean=334,485,246 on first run, vs the next-highest region's
    # tens of thousands). b73_n_gaps.bed already flags whether a *region* touches a gap
    # (dist_to_n_gap); this excludes the affected *tiles* from the occurrence statistic
    # itself so one gap-straddling tile can't dominate a region's mean/variance.
    n_masked_keys = set()
    with open(tiles_fa) as f:
        key, seq = None, []
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if key is not None and "N" in "".join(seq).upper():
                    n_masked_keys.add(key)
                chrom, rest = line[1:].split(":")
                s1, e1 = (int(x) for x in rest.split("-"))
                key = region_key(chrom, s1 - 1, e1)
                seq = []
            else:
                seq.append(line)
        if key is not None and "N" in "".join(seq).upper():
            n_masked_keys.add(key)
    if n_masked_keys:
        print(f"{len(n_masked_keys)} tiles ({len(n_masked_keys)/len(tile_owners):.3%}) contain N "
              f"(assembly gap) -- excluded from occurrence stats")

    mem_out = out_dir / "occ_tiles.mem.tsv"
    mem_err = out_dir / "occ_tiles.mem.log"
    with open(mem_out, "w") as out_f, open(mem_err, "w") as err_f:
        subprocess.run([RB3, "mem", "-l", "19", "-t", "8", IDX, str(tiles_fa)],
                        stdout=out_f, stderr=err_f, check=True)

    # samtools faidx prints headers as "chrom:1based_s-1based_e" -- convert back to our
    # 0-based key. Aggregate multiple MEM lines per tile: keep the longest-span match
    # (most representative single alignment for that tile), tie-break by larger occ.
    tile_occ = {}
    best_span = {}
    with open(mem_out) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            name, ms, me, occ = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
            chrom, rest = name.split(":")
            s1, e1 = (int(x) for x in rest.split("-"))
            key = region_key(chrom, s1 - 1, e1)
            if key in n_masked_keys:
                continue
            span = me - ms
            if key not in best_span or span > best_span[key] or (span == best_span[key] and occ > tile_occ[key]):
                best_span[key] = span
                tile_occ[key] = occ
    missing = [k for k in tile_owners if k not in tile_occ and k not in n_masked_keys]
    if missing:
        print(f"WARNING: {len(missing)} tiles had no MEM >= 19bp (below min MEM length); "
              f"excluded from stats, e.g. {missing[:3]}")

    def stats_for(occs):
        if not occs:
            return None
        occs_sorted = sorted(occs)
        n = len(occs_sorted)
        mean = st.mean(occs_sorted)
        out = {
            "n_tiles": n,
            "occ_mean": round(mean, 3),
            "occ_median": occs_sorted[n // 2],
            "occ_var": round(st.pvariance(occs_sorted), 3) if n > 1 else 0.0,
            "occ_cv": round((st.pstdev(occs_sorted) / mean), 4) if n > 1 and mean > 0 else None,
            "occ_p10": occs_sorted[max(0, int(n * 0.10))],
            "occ_p90": occs_sorted[min(n - 1, int(n * 0.90))],
            "frac_le2": round(sum(1 for x in occs_sorted if x <= 2) / n, 4),
            "frac_ge100": round(sum(1 for x in occs_sorted if x >= 100) / n, 4),
        }
        out["p90_p10_ratio"] = (out["occ_p90"] / out["occ_p10"]) if out["occ_p10"] > 0 else None
        return out

    results = []
    for i, (r, kind) in enumerate(all_regions):
        keys = region_tile_keys[i]
        occs = [tile_occ[k] for k in keys if k in tile_occ]
        n_masked = sum(1 for k in keys if k in n_masked_keys)
        s = stats_for(occs)
        row = {"chrom": r["chrom"], "start": r["start"], "end": r["end"], "width": r["width"],
               "kind": kind, "true_founder": r.get("true_founder"), "n_tiles_n_masked": n_masked}
        row.update(s if s else {"n_tiles": 0})
        row["_tile_occs"] = occs  # kept for step 2 reuse (jump-rate / autocorrelation)
        results.append(row)

    out_json = out_dir / "region_occ_stats.json"
    out_json.write_text(json.dumps(results))
    zero = [r for r in results if r.get("n_tiles", 0) == 0]
    print(f"wrote {out_json} ({len(results)} regions, {len(zero)} with zero usable tiles "
          f"-- fully inside an N-gap, dropped from the comparison below)")
    for r in zero:
        print(f"  dropped: {r['kind']} {r['chrom']}:{r['start']}-{r['end']}")

    err_vals = {k: [r[k] for r in results if r["kind"] == "error" and r.get("n_tiles", 0) > 0]
                for k in ("occ_mean", "occ_cv", "frac_ge100")}
    bg_vals = {k: [r[k] for r in results if r["kind"] == "background" and r.get("n_tiles", 0) > 0]
               for k in ("occ_mean", "occ_cv", "frac_ge100")}
    print(f"\nerror (n={len(err_vals['occ_mean'])}) vs background (n={len(bg_vals['occ_mean'])}):")
    try:
        from scipy import stats as sps
        for k in ("occ_mean", "occ_cv", "frac_ge100"):
            e, b = [v for v in err_vals[k] if v is not None], [v for v in bg_vals[k] if v is not None]
            u, p = sps.mannwhitneyu(e, b, alternative="two-sided")
            print(f"  {k}: error median={st.median(e):.4g} background median={st.median(b):.4g} "
                  f"Mann-Whitney p={p:.4g}")
    except ImportError:
        print("  (scipy unavailable -- rerun under ml-impute-env for the significance test)")


if __name__ == "__main__":
    main()
