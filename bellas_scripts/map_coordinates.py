import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Replace Start/Stop in a map file with coordinates from a bed file")
parser.add_argument("map_file", help="Map file with columns: Marker,Chr,Start,Stop,Bin size (Mb),Genetic map (cM),Family,Pop")
parser.add_argument("bed_file", help="BED file with columns: Chr,Start,Stop")
parser.add_argument("output_file", help="Output map file path")
args = parser.parse_args()

map_df = pd.read_csv(args.map_file)
bed_df = pd.read_csv(args.bed_file, sep="\t", header=None, names=["Chr", "Start", "Stop", "N/A"])

if len(map_df) != len(bed_df):
    raise ValueError(f"Row count mismatch: map has {len(map_df)} rows, bed has {len(bed_df)} rows")

print(map_df.head())
print(bed_df.head())
print(bed_df["Start"].head)
print(bed_df["Stop"].head)


map_df["Start"] = bed_df["Start"].values
map_df["Stop"] = bed_df["Stop"].values

map_df.to_csv(args.output_file, sep="\t", index=False)
print(f"Written {len(map_df)} rows to {args.output_file}")
