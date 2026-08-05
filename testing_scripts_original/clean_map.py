#!/usr/bin/env python3
"""
Clean a genetic map file (tab-separated, no header, columns: pos  chr  cM).

Steps per chromosome:
  1. Sort by pos ascending, then cM ascending
  2. Remove rows with negative cM
  3. Remove rows where pos increases but cM decreases (monotonicity filter)

Outputs:
  <out_dir>/<chr>_cleaned.map  — one file per chromosome  (for impute5)
  <out_dir>/combined_cleaned.map — all chromosomes combined (for minimac4)
"""
import os
import sys
import argparse
import pandas as pd


def enforce_monotonic_cM(df):
    """Keep only rows where cM is non-decreasing as pos increases."""
    keep = []
    max_cM = float("-inf")
    for cM_val in df["cM"]:
        if cM_val >= max_cM:
            keep.append(True)
            max_cM = cM_val
        else:
            keep.append(False)
    return df[keep].reset_index(drop=True)


def clean_chr_df(df):
    df = df.sort_values(["pos", "cM"]).reset_index(drop=True)
    df = df[df["cM"] >= 0].reset_index(drop=True)
    df = enforce_monotonic_cM(df)
    return df


def normalize_chr(val):
    s = str(val)
    return s if s.startswith("chr") else "chr" + s


def chr_sort_key(chrom):
    num = chrom.replace("chr", "")
    try:
        return (0, int(num))
    except ValueError:
        return (1, num)


def main():
    parser = argparse.ArgumentParser(
        description="Clean and split a genetic map file by chromosome"
    )
    parser.add_argument("map_file", help="Input map (tab-sep, no header): pos  chr  cM")
    parser.add_argument("out_dir", help="Output directory for per-chr and combined maps")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.map_file, sep="\t", header=None, names=["pos", "chr", "cM"],
                     dtype={"pos": int, "chr": str, "cM": float})

    df["chr"] = df["chr"].apply(normalize_chr)

    chromosomes = sorted(df["chr"].unique(), key=chr_sort_key)

    all_cleaned = []
    for chrom in chromosomes:
        chr_df = df[df["chr"] == chrom].copy()
        n_before = len(chr_df)

        cleaned = clean_chr_df(chr_df)
        n_after = len(cleaned)

        print(f"{chrom}: {n_before} rows -> {n_after} kept ({n_before - n_after} removed)")

        out_path = os.path.join(args.out_dir, f"{chrom}_cleaned.map")
        cleaned.to_csv(out_path, sep="\t", header=False, index=False)

        all_cleaned.append(cleaned)

    combined = pd.concat(all_cleaned, ignore_index=True)
    combined_path = os.path.join(args.out_dir, "combined_cleaned.map")
    combined.to_csv(combined_path, sep="\t", header=False, index=False)

    total_in = len(df)
    total_out = len(combined)
    print(f"\nTotal: {total_in} -> {total_out} rows ({total_in - total_out} removed)")
    print(f"Per-chr files and combined map written to: {args.out_dir}")


if __name__ == "__main__":
    main()
