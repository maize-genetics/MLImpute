import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Standard map to minimac map format")
parser.add_argument("map_file", help="Map file with columns: Marker,Chr,Start,Stop,Bin size (Mb),Genetic map (cM),Family,Pop")
parser.add_argument("output_file", help="Output map file path")
args = parser.parse_args()

map_df = pd.read_csv(args.map_file, sep="\t")

map_df["Position"] = (((map_df["Stop"] - map_df["Start"]) / 2) + map_df["Start"]).astype(int)

out_df = pd.DataFrame({"Chr": "chr" + map_df["Chr"].astype(str), "N/A": ".", "cM": map_df["Genetic map (cM)"], "Position": map_df["Position"]})

out_df.to_csv(args.output_file, sep="\t", index=False)
print(f"Written {len(map_df)} rows to {args.output_file}")