import numpy as np
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-parents", type=int, default=24, help="number of parents")
    parser.add_argument("--window-size", type=int, default=6144, help="window length")
    parser.add_argument("--step-size", type=int, default=1536, help="stride")
    parser.add_argument("--keyfile", type=str, required=True, help="directory containing the npy files")
    parser.add_argument("--output", "-o", type=str, required=True, help="output keyfile")

    args = parser.parse_args()
    return args


def bins_to_idx(labels_binned):
    return [idx+1 for idx in range(labels_binned.shape[0] - 1) if labels_binned[idx] != labels_binned[idx+1]]

def num_crossovers(arr, start, length, parent_idx):
    return len(bins_to_idx(arr[start:(start+length), parent_idx]))


args = parse_args()

windows = []

keyfile = pd.read_csv(args.keyfile, sep="\t")

for idx, row in keyfile.iterrows():
    x = np.load(row["path"])

    num_windows = (x.shape[0] - args.window_size) // args.step_size
    windows.extend([(idx, idy) for idy in range(num_windows) if
                    num_crossovers(x, (idy*args.step_size), args.window_size, args.num_parents) > 5])



df = pd.DataFrame(windows, columns=["file_idx", "pos_idx"])

df.to_csv(args.output, sep="\t", index=False)



