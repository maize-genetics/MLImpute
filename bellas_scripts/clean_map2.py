#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd


def enforce_monotonic_cM(df, col="cM"):
    """Keep only rows where cM is non-decreasing as position increases."""
    keep = []
    max_cM = float("-inf")
    for val in df[col]:
        if val >= max_cM:
            keep.append(True)
            max_cM = val
        else:
            keep.append(False)
    return df[keep].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Clean and split a genetic map file by chromosome"
    )
    parser.add_argument("map_file", help="Input map (tab-sep, no header): pos  chr  cM")
    parser.add_argument("out_file", help="Output directory for per-chr and combined maps")
    args = parser.parse_args()

    df = pd.read_csv(args.map_file, sep="\t", header=0,
                     names=["Marker", "Chr", "Start", "Stop", "Bin size (Mb)", "Genetic map (cM)", "Family", "Pop"])
    
    df["Position"] = (((df["Stop"].astype(int) - df["Start"].astype(int)) / 2) + df["Start"].astype(int)).astype(int)

    # 1. sort by ascending position, then ascending cM
    df = df.sort_values(["Position", "Genetic map (cM)"]).reset_index(drop=True)

    # 2. remove duplicate rows
    df = df.drop_duplicates(subset=["Position"]).reset_index(drop=True)

    # 3. remove rows where cM < 0
    df = df[df["Genetic map (cM)"] >= 0].reset_index(drop=True)

    # 4. per chromosome: drop rows where cM decreases as position increases
    df = (
        df.groupby("Chr", sort=False, group_keys=False)
        .apply(lambda g: enforce_monotonic_cM(g.sort_values("Position"), col="Genetic map (cM)"))
        .reset_index(drop=True)
    )

    print(df.head())
    
    df.to_csv(args.out_file, sep='\t', index=False)



if __name__ == "__main__":
    main()
