import pandas as pd
import argparse
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="takes a ps4g file and the sample_parents.txt output from PHG HMM and converts it into a numerical numpy array to be used in downstream visualization analysis")
    parser.add_argument("--input-bed", type=str, required=True, help="Input bed file")
    parser.add_argument("--ps4g", type=str, required=True, help="PS4G file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--bin-size", type=int, default=256, help="Bin size")
    parser.add_argument("--diploid", action='store_true', help="Diploid predictions")
    args = parser.parse_args()

    # BED
    df = pd.read_csv(args.input_bed, sep="\t")

    for c in df.columns[3:]:
        df[c] = df[c].astype(str).str.replace(r":\d+$", "", regex=True)

    df["start"] = (df["start"] // args.bin_size).astype(int)
    df["end"]   = (df["end"] // args.bin_size).astype(int)

    parent_col_1 = df.columns[3]
    if args.diploid:
        parent_col_2 = df.columns[4]

    # PS4G
    ps4g = pd.read_csv(args.ps4g, sep="\t", comment="#")
    sample = os.path.basename(args.ps4g).split(".")[0]

    with open(args.ps4g, 'r') as file:
        comments = [line for line in file if line.startswith('#')]
    gamete_data = []
    for line in comments:
        line = line.strip()
        if line.startswith("#") and ":" in line and "\t" in line:
            # Example line: "#B73:0\t1\t10730006"
            line = line[1:]  # Remove leading "#"
            gamete_full, idx, count = line.split("\t")
            gamete_name = gamete_full.split(":")[0]
            gamete_data.append({
                "gamete": gamete_name,
                "gamete_index": int(idx),
            })

    name_to_index = {entry["gamete"]: entry["gamete_index"] for entry in gamete_data}
    print(name_to_index)

    # unique positions
    unique_pos = (
        ps4g[["refContig", "refPosBinned"]]
        .drop_duplicates()
        .sort_values(["refContig", "refPosBinned"])
    )
    print(unique_pos)

    for chrom, pos_df in unique_pos.groupby("refContig"):

        bed_chr = df[df["chrom"] == chrom]
        if bed_chr.empty:
            continue

        pos = pos_df["refPosBinned"].values
        if args.diploid:
            path = np.full((len(pos), 2), -1, dtype=int)
            print(path.shape)
        else:
            path = np.full(len(pos), -1, dtype=int)

        for row in bed_chr.itertuples():
            mask = (pos >= row.start) & (pos <= row.end)
            parent_1 = getattr(row, parent_col_1)
            if args.diploid:
                parent_2 = getattr(row, parent_col_2)
                path[mask, 0] = name_to_index[parent_1]
                path[mask, 1] = name_to_index[parent_2]
            else:
                path[mask] = name_to_index[parent_1]

        os.makedirs(args.output_dir, exist_ok=True)
        np.save(f"{args.output_dir}/{sample}_{chrom}.npy", path)
        print(path[:10])

if __name__ == "__main__":
    main()