# This script selects a list of windows to be used as a dataset for one of the training/inference scripts
# Its main use is to downsample windows that have no crossover points

import numpy as np
import pandas as pd
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--keyfile", type=str, required=True, help="directory containing the npy files")
    parser.add_argument("--output", "-o", type=str, required=True, help="output keyfile")
    parser.add_argument("--num-parents", type=int, default=24, help="number of parents")
    parser.add_argument("--step-size", type=int, default=128, help="step size between window starts")
    parser.add_argument("--input-len", type=int, default=256, help="length of windows")
    parser.add_argument("--padding", type=int, default=0, help="padding length: defines a 'border crossover' vs a 'core crossover'")
    parser.add_argument("--border-retain-rate", type=float, default=0.5, help="rate at which to retain windows with only border crossovers")
    parser.add_argument("--no-crossover-rate", type=float, default=0.02, help="rate at which to retain windows with no crossovers")

    args = parser.parse_args()
    return args


def bins_to_idx(labels_binned) -> list:
    return [idx+1 for idx in range(labels_binned.shape[0] - 1) if labels_binned[idx] != labels_binned[idx+1]]

def cat_crossover(arr, start, length, parent_idx):
    crosses = bins_to_idx(arr[start:(start + length), parent_idx])

    border_cross = len([y for y in crosses if not 8 <= y < 24])
    target_cross = len([y for y in crosses if 8 <= y < 24])

    return target_cross, border_cross

args = parse_args()

windows = []

keyfile = pd.read_csv(args.keyfile, sep="\t")


for idx, row in keyfile.iterrows():
    x = np.load(row["path"])

    num_windows = (x.shape[0] - args.input_len) // args.step_size

    crosses_per_window = [cat_crossover(x, (idy*args.step_size), args.input_len, args.num_parents)
                          for idy in range(num_windows)]

    windows.extend([(idx, idy) for idy in range(num_windows) if
                   (crosses_per_window[idy][0] > 0) or
                   (crosses_per_window[idy][0] == 0 and crosses_per_window[idy][1] > 0 and random.random() < args.border_retain_rate) or
                   (crosses_per_window[idy][0] == 0 and crosses_per_window[idy][1] == 0 and random.random() < args.no_crossover_rate)])



df = pd.DataFrame(windows, columns=["file_idx", "pos_idx"])

df.to_csv(args.output, sep="\t", index=False)



