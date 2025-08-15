import pandas as pd
import argparse
import numpy as np
from ps4g_io.ps4g import load_ps4g_file, build_index_lookup, collapse_ps4g


def count_parents(bed_df):
    parents = bed_df["parent1"] + bed_df["parent2"]
    return parents.nunique()


def count_crossovers(bed_df):
    cols = ["parent1", "parent2"]
    diffs = bed_df[cols].ne(bed_df[cols].shift()).any(axis=1)
    return int(diffs.iloc[1:].sum())  # skip first row

# TODO: make diploid
def compute_accuracy(bed_df, ps4g_file, collapsed=True):
    ps4g = load_ps4g_file(ps4g_file)
    index_array = build_index_lookup(ps4g_file)
    name_to_idx = {name: i for i, name in enumerate(index_array)}
    if collapsed:
        num_classes = len(index_array)
        unique_positions = ps4g['pos'].unique()
        _, ps4g = collapse_ps4g(num_classes, ps4g, unique_positions)
    parents = bed_df['parent1'].map(name_to_idx)
    gametes = ps4g['gameteSet'].reindex(parents.index)
    hits = parents.combine(gametes, lambda p, g: (p in g) if isinstance(g, (set, list, tuple, np.ndarray, pd.Series)) else p == g).fillna(False)
    accuracy = hits.mean()
    return f"{accuracy:.2%}"


def collapse_and_aggregate(bed_df):
    def _mode(s: pd.Series):
        m = s.mode(dropna=True)
        return m.iloc[0] if len(m) else np.nan

    out = (
        bed_df
        .groupby(["chrom_idx", "pos"], as_index=False, sort=True)
        .agg(parent1=("parent1", _mode),
             parent2=("parent2", _mode),
             n=("parent1", "size"))
        .sort_values(["chrom_idx", "pos"], kind="mergesort")
        .reset_index(drop=True)
    )
    return out


def parent_contribution(bed_df):
    # next boundary within each chromosome
    bed_df["next_pos"] = bed_df.groupby('chrom_idx', sort=False)['pos'].shift(-1)

    # segment length
    bed_df["length"] = bed_df["next_pos"] - bed_df['pos']

    # keep only positive lengths
    bed_df = bed_df[bed_df["length"] > 0]

    # sum by parent and normalize
    by_parent = bed_df.groupby('parent1', sort=False)["length"].sum()
    total = by_parent.sum()

    if total == 0:
        # no measurable segments
        return by_parent * 0.0

    frac = by_parent / total
    return (frac * 100).round(2)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bed-file', type=str, help='bed file')
    parser.add_argument("--ps4g-file", type=str, default=None)
    parser.add_argument("--collapsed", action='store_true')
    args = parser.parse_args()

    bed_df = pd.read_csv(args.bed_file, sep='\t', header=0) # input full bed file (not ranges)

    if args.collapsed:
        bed_df = collapse_and_aggregate(bed_df)

    num_parents = count_parents(bed_df)
    print("Number of Contributing Parents: ", num_parents)

    num_crossovers = count_crossovers(bed_df)
    print("Number of Crossovers: ", num_crossovers)

    accuracy = compute_accuracy(bed_df, args.ps4g_file, collapsed=args.collapsed)
    print("Accuracy: ", accuracy)

    percent = parent_contribution(bed_df)
    print(percent.map(lambda x: f"{x:.2f}%"))

if __name__ == '__main__':
    main()