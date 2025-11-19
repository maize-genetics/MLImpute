import pandas as pd
import subprocess
import os
import argparse
from python.cross.chrom_lengths import chrom_lengths_dicts

"""
create a bed file to track fasta sequence to assembly
columns = fa_chr, fa_start, fa_end, parent_chr, parent_start, parent_end, parent
"""
def build_fasta_keys(parents, founder):
    fasta_df = pd.DataFrame(
columns=["fa_chr", "fa_start", "fa_end", "parent_chr", "parent_start", "parent_end", "parent", "seg_len"])

    founder_df = pd.DataFrame(columns=["ref_chr", "ref_start", "ref_end", "parent_chr", "parent_start", "parent_end", "founder"])
    key = []

    for parent in parents: # for each parent, subset founder and aggregate
        parent_df = pd.read_csv(f"{parent}_key.bed", sep="\t", header=None,
                                names=["parent_chr", "parent_start", "parent_end", "ref_chr", "ref_start", "ref_end", "founder"])
        parent_subset = parent_df[parent_df["founder"] == int(founder)]
        founder_df = pd.concat([founder_df, parent_subset], ignore_index=True)
        key.extend([parent]*len(parent_subset))

    founder_df["parent"] = key
    # sort based on ref_chr, ref_start, ref_end
    founder_df = founder_df.sort_values(by=["ref_chr", "ref_start", "ref_end"], ascending=[True, True, True])
    fasta_df["fa_chr"] = founder_df["ref_chr"]
    fasta_df["seg_len"] = (founder_df["parent_end"] - founder_df["parent_start"]).astype(int)
    fasta_df["fa_end"] = fasta_df.groupby("fa_chr")["seg_len"].cumsum()
    fasta_df["fa_start"] = fasta_df["fa_end"] - fasta_df["seg_len"]
    fasta_df = fasta_df.drop(columns=["seg_len"])
    fasta_df["parent_chr"] = founder_df["parent_chr"]
    fasta_df["parent_start"] = founder_df["parent_start"]
    fasta_df["parent_end"] = founder_df["parent_end"]
    fasta_df["parent"] = founder_df["parent"]

    fasta_df.to_csv(f"{founder}_key.bed", sep="\t", index=False, header=False)

"""
edit assembly coordinates to ensure intervals cover all assembly sequence
"""
def adjust_coords(df, length):
    df.loc[df.index[0], "parent_start"] = 0
    end = df.loc[df.index[0], "parent_end"]

    # iterate from second row onward
    for i in df.index[1:]:
        if df.at[i, "parent_start"] != end:
            df.at[i, "parent_start"] = end
        end = df.at[i, "parent_end"]

    # set last end to full chromosome length
    df.loc[df.index[-1], "parent_end"] = length

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assembly-list", type=str, help="file containing full file paths and names for assembly fastas")
    parser.add_argument("--chain-dir", type=str, help="chain file directory")
    args = parser.parse_args()

    assembly_founder_paths = []
    assembly_founders = []

    with open(args.assembly_list) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                assembly_founder_paths.append(parts[0])
                assembly_founders.append(parts[1])

    founder_chroms = chrom_lengths_dicts(assembly_founder_paths, exclude_scaffolds=True)

    # add duplicate columns to refkey files
    for founder in assembly_founders:
        refkey = f"{founder}_refkey.bed"
        out = f"{founder}_refkey_temp.bed"
        df = pd.read_csv(refkey, sep="\t", header=None)
        df_new = df[[0, 1, 2, 0, 1, 2, 3]]  # zero-based indexing
        df_new.to_csv(out, sep="\t", header=False, index=False)

    for founder in assembly_founders:
        # File paths
        temp_bed = f"{founder}_refkey_temp.bed"
        chain_file = os.path.join(args.chain_dir, f"{founder}.chain")
        out_bed = f"{founder}_key.bed"

        # Run CrossMap
        subprocess.run([
            "CrossMap",
            "bed",
            chain_file,
            temp_bed,
            out_bed
        ], check=True)

        # Remove temporary file
        os.remove(temp_bed)

    # sort by parent coords and fill in missing chunks
    for founder in assembly_founders:
        key = f"{founder}_key.bed"
        key_df = pd.read_csv(key, sep="\t", header=None, names=["parent_chr", "parent_start", "parent_end", "ref_chr", "ref_start", "ref_end", "founder"])

        # sort once
        sorted_df = key_df.sort_values(
            by=["parent_chr", "parent_start", "parent_end"],
            ascending=[True, True, True]
        )

        # adjust per chromosome, collect parts
        adjusted_parts = []
        for c, length in founder_chroms[founder].items():
            chunk = sorted_df[sorted_df["parent_chr"] == c]
            if chunk.empty:
                continue
            adj = adjust_coords(chunk, int(length))
            # keep the zero-length guard (belt & suspenders)
            adj = adj[adj["parent_start"] != adj["parent_end"]]
            adjusted_parts.append(adj)

        if adjusted_parts:
            out_df = pd.concat(adjusted_parts, ignore_index=True)
        else:
            out_df = sorted_df

        out_df.to_csv(f"{founder}_key.bed", sep="\t", index=False, header=False)

    for i in range(len(assembly_founders)):
        build_fasta_keys(assembly_founders, i)