#!/usr/bin/env python
"""
Build the founder x B73-position touchdown matrix directly from ropebwt3's
precomputed .lift file (see parse_lift_file.py for the format) -- no new
alignment or tiling. Every point in the lift file is a (carrier_pos) ->
(B73_pos) single-copy-per-taxon anchor found when the index was built.

Binning B73 positions into fixed-width bins per chromosome and counting
anchor points per (founder, bin) gives a genome-wide "where does each
founder's assembly touch down cleanly on B73" density map. Two readings:
  - A bin that's cold across nearly ALL founders -> the region itself is
    too repetitive/complex for the lift-builder's single-copy filter
    (max_occ) to find anchors there at all -- a real, repeat-driven
    ambiguous region (like chr5:~200-202.5Mb found earlier this session).
  - A bin that's cold for ONE founder while its neighbors and other
    founders are fine -> that founder's own assembly specifically lacks
    (or fails to place) sequence there -- the "poor local assembly"
    signal being asked for.
"""
import argparse
import gzip
import json
import struct
from pathlib import Path

import numpy as np

LIFTPT_DTYPE = np.dtype([("cpos", "<i8"), ("rpos", "<i8"), ("csid", "<i4"), ("rsid", "<i4")])
CHROMS = [f"chr{i}" for i in range(1, 11)]


def load_seq_names(len_gz_path):
    names = []
    with gzip.open(len_gz_path, "rt") as f:
        for line in f:
            name, _ = line.rstrip("\n").split("\t")
            names.append(name)
    return names


def read_lift_points(lift_path):
    with open(lift_path, "rb") as f:
        magic = f.read(5)
        assert magic == b"LIFT\x01", f"bad magic {magic!r}"
        n_seq, n_pt = struct.unpack("<qq", f.read(16))
        f.read(8 * (n_seq + 1))  # off[] -- not needed, we scan all points directly
        pts = np.frombuffer(f.read(LIFTPT_DTYPE.itemsize * n_pt), dtype=LIFTPT_DTYPE, count=n_pt)
    return n_seq, pts


def load_b73_chrom_lengths(fai_path):
    lengths = {}
    with open(fai_path) as f:
        for line in f:
            name, length = line.split("\t")[:2]
            if name.startswith("chr") and name[3:].isdigit():
                lengths[name] = int(length)
    return lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx-prefix", default="/workdir/zrm22/HackathonJun2026/ropebwt_refMap/rope_bwt_index_v2/maizeFastaIndex_SampleContig_v2")
    ap.add_argument("--b73-fai", default="/workdir/shared_files/grits_crf_evaluation/index_asms/maize_v2/B73.fa.fai")
    ap.add_argument("--bin-size", type=int, default=1_000_000)
    ap.add_argument("--out-json", required=True)
    cli = ap.parse_args()

    names = load_seq_names(cli.idx_prefix + ".fmd.len.gz")
    n_seq, pts = read_lift_points(cli.idx_prefix + ".lift")
    print(f"{len(pts):,} anchor points, {n_seq:,} sequences")

    chrom_lengths = load_b73_chrom_lengths(cli.b73_fai)
    n_bins_per_chrom = {c: (chrom_lengths[c] // cli.bin_size) + 1 for c in CHROMS}
    bin_offset = {}  # global bin index offset per chromosome, for one flat x-axis
    off = 0
    for c in CHROMS:
        bin_offset[c] = off
        off += n_bins_per_chrom[c]
    total_bins = off
    print(f"total B73 bins @ {cli.bin_size:,}bp: {total_bins}")

    # csid -> (founder, is_founder_chrom_i.e.not_scaffold) ; rsid -> (chrom, global_bin) or None
    founder_of_sid = [None] * n_seq
    for i, nm in enumerate(names):
        founder_of_sid[i] = nm.split("_", 1)[0]

    rsid_to_globalbin = np.full(n_seq, -1, dtype=np.int64)
    for i, nm in enumerate(names):
        if not nm.startswith("B73_"):
            continue
        rest = nm[len("B73_"):]
        if rest in chrom_lengths:
            # placeholder; actual per-point bin computed from rpos below, this only
            # flags which sids ARE a usable B73 chromosome and records which chrom
            rsid_to_globalbin[i] = CHROMS.index(rest)
    is_b73_chrom_sid = rsid_to_globalbin >= 0

    founders = sorted(set(founder_of_sid) - {"B73"})
    founder_idx = {f: i for i, f in enumerate(founders)}
    print(f"{len(founders)} founders (excluding B73): {founders}")

    matrix = np.zeros((len(founders), total_bins), dtype=np.int64)
    founder_total_anchors = np.zeros(len(founders), dtype=np.int64)

    csid = pts["csid"]
    rsid = pts["rsid"]
    rpos = pts["rpos"]

    valid_ref = is_b73_chrom_sid[rsid]
    carrier_founder_names = np.array([founder_of_sid[s] for s in csid])
    not_b73_carrier = carrier_founder_names != "B73"
    keep = valid_ref & not_b73_carrier
    print(f"{keep.sum():,} / {len(pts):,} points kept (B73-chrom target, non-B73 carrier)")

    kept_founder = carrier_founder_names[keep]
    kept_chrom_idx = rsid_to_globalbin[rsid[keep]]
    kept_rpos = rpos[keep]
    kept_bin_in_chrom = kept_rpos // cli.bin_size

    chrom_names_arr = np.array(CHROMS)
    kept_offsets = np.array([bin_offset[c] for c in chrom_names_arr[kept_chrom_idx]])
    kept_global_bin = kept_offsets + kept_bin_in_chrom

    for f in founders:
        fi = founder_idx[f]
        mask = kept_founder == f
        founder_total_anchors[fi] = mask.sum()
        bc = np.bincount(kept_global_bin[mask], minlength=total_bins)
        matrix[fi, :] = bc[:total_bins]

    out = {
        "bin_size": cli.bin_size,
        "founders": founders,
        "chroms": CHROMS,
        "chrom_lengths": chrom_lengths,
        "bin_offset": bin_offset,
        "n_bins_per_chrom": n_bins_per_chrom,
        "total_bins": total_bins,
        "founder_total_anchors": founder_total_anchors.tolist(),
        "matrix": matrix.tolist(),
    }
    Path(cli.out_json).write_text(json.dumps(out))
    print(f"wrote {cli.out_json}")
    print(f"matrix shape: {matrix.shape}, total anchors plotted: {matrix.sum():,}")
    print("founder total anchors (min/median/max):",
          founder_total_anchors.min(), int(np.median(founder_total_anchors)), founder_total_anchors.max())


if __name__ == "__main__":
    main()
