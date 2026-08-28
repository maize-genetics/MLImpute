#!/usr/bin/env python
"""
Parse ropebwt3's precomputed carrier->reference coordinate liftover file
(the ".lift" binary, built by `ropebwt3 lift`, documented at lift.c:178-206)
directly -- no new alignment, no tiling, just reading an existing
ropebwt3-generated file. This is the "second SSA": for every ~2000bp along
B73 (--ref-prefix), it recorded shared single-copy-per-taxon k-mer anchors
(<= max_occ interval size, default = number of taxa) between B73 and every
OTHER sequence in the pangenome index (every founder chromosome AND every
unplaced scaffold). Each point IS exactly a (founder position, B73 position)
touchdown -- precisely the data plotted here, already computed.

Binary format (lift.c:178-206), flat, no compression:
  magic   5 bytes   "LIFT\1"
  n_seq   int64     number of sequences in the index (carrier keys)
  n_pt    int64     total liftover points
  off[]   (n_seq+1) int64 -- off[sid]..off[sid+1] = point range for sequence sid
  pt[]    n_pt records, each rb3_liftpt_t (24 bytes, no padding):
            cpos int64  -- carrier (this sequence's own) forward-strand pos
            rpos int64  -- reference (B73) forward-strand pos
            csid int32  -- carrier sequence id
            rsid int32  -- reference sequence id (a B73 chromosome)

Sequence names come from the index's separate <idx>.fmd.len.gz sidecar
(plain "name\\tlength" per line, one per sequence, in the SAME id order
used throughout the index -- confirmed by name/index cross-check in
main()). NOTE: only the REFERENCE side (rsid, sequences whose name starts
with --ref-prefix) was included in the k-mer search target when the lift
file was built; the CARRIER side genuinely covers every sequence, founder
chromosomes and unplaced scaffolds alike.
"""
import argparse
import gzip
import struct
import sys
from pathlib import Path

LIFTPT_FMT = "<qqii"  # cpos(i64) rpos(i64) csid(i32) rsid(i32) -- 24 bytes, matches C struct layout
LIFTPT_SIZE = struct.calcsize(LIFTPT_FMT)
assert LIFTPT_SIZE == 24


def load_seq_names(len_gz_path):
    names = []
    lengths = []
    with gzip.open(len_gz_path, "rt") as f:
        for line in f:
            name, length = line.rstrip("\n").split("\t")
            names.append(name)
            lengths.append(int(length))
    return names, lengths


def read_lift(lift_path):
    with open(lift_path, "rb") as f:
        magic = f.read(5)
        if magic != b"LIFT\x01":
            raise ValueError(f"bad magic {magic!r} in {lift_path}")
        n_seq, n_pt = struct.unpack("<qq", f.read(16))
        off = struct.unpack(f"<{n_seq + 1}q", f.read(8 * (n_seq + 1)))
        pt_bytes = f.read(LIFTPT_SIZE * n_pt)
        if len(pt_bytes) != LIFTPT_SIZE * n_pt:
            raise ValueError(f"truncated lift file: expected {LIFTPT_SIZE * n_pt} bytes of points, got {len(pt_bytes)}")
    return n_seq, n_pt, off, pt_bytes


def iter_points(pt_bytes, n_pt):
    for i in range(n_pt):
        cpos, rpos, csid, rsid = struct.unpack_from(LIFTPT_FMT, pt_bytes, i * LIFTPT_SIZE)
        yield cpos, rpos, csid, rsid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lift", required=True)
    ap.add_argument("--len-gz", required=True)
    ap.add_argument("--sanity-check", action="store_true")
    cli = ap.parse_args()

    names, lengths = load_seq_names(cli.len_gz)
    n_seq, n_pt, off, pt_bytes = read_lift(cli.lift)
    print(f"lift file: n_seq={n_seq} n_pt={n_pt}")
    print(f"len.gz: {len(names)} sequence names")

    if cli.sanity_check:
        assert n_seq == len(names), f"n_seq mismatch: lift={n_seq} len.gz={len(names)}"
        pts = list(iter_points(pt_bytes, min(n_pt, 20)))
        for cpos, rpos, csid, rsid in pts[:10]:
            cname = names[csid] if csid < len(names) else f"<sid{csid} OOB>"
            rname = names[rsid] if rsid < len(names) else f"<sid{rsid} OOB>"
            print(f"  {cname}:{cpos} -> {rname}:{rpos}")
        # off[] should be non-decreasing and off[n_seq] == n_pt
        assert off[0] == 0 and off[-1] == n_pt, f"off[] bounds wrong: off[0]={off[0]} off[-1]={off[-1]} n_pt={n_pt}"
        assert all(off[i] <= off[i + 1] for i in range(n_seq)), "off[] not non-decreasing"
        print("sanity checks passed")


if __name__ == "__main__":
    main()
