import pandas as pd
import argparse
import numpy as np


def count_parents(bed_df):
    parents = bed_df["parent1"] + bed_df["parent2"]
    return parents.nunique()


def count_crossovers(bed_df):
    cols = ["parent1", "parent2"]
    diffs = bed_df[cols].ne(bed_df[cols].shift()).any(axis=1)
    return int(diffs.iloc[1:].sum())  # skip first row

# TODO: calculate accuracy
def compute_accuracy(bed_df, matrix):
    accuracy = 0
    return accuracy


# TODO: collapse/aggregate duplicate positions
def collapse_and_aggregate(args):
    bed_df = pd.read_csv(args.bed_file, sep='\t', header=0)

    return bed_df


# TODO: calculate base pair percentages for each parent
def parent_contribution(bed_df):
    parents = bed_df["parent1"].unique()
    percentages = 0
    return parents, percentages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bed-file', type=str, help='bed file')
    parser.add_argument("--matrix", default=None)
    args = parser.parse_args()

    bed_df = pd.read_csv(args.bed_file, sep='\t', header=0) # input full bed file (not ranges)
    matrix = np.load(args.matrix, allow_pickle=True)

    num_parents = count_parents(bed_df)
    print("Number of Contributing Parents: ", num_parents)

    num_crossovers = count_crossovers(bed_df)
    print("Number of Crossovers: ", num_crossovers)

    accuracy = compute_accuracy(bed_df, matrix)
    print("Accuracy: ", accuracy)

    parent, percent = parent_contribution(bed_df)
    print(parent, percent)

if __name__ == '__main__':
    main()