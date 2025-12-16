import numpy as np
import pandas as pd
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--keyfile", type=str, required=True, help="directory containing the npy files")
    parser.add_argument("--output", "-o", type=str, required=True, help="output keyfile")

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

total_windows = 0
total_pos = 0
total_border = 0
total_multi = 0

for idx, row in keyfile.iterrows():
    x = np.load(row["path"])
    print(idx)

    num_windows = (x.shape[0] - args.window_size) // args.step_size

    crosses_per_window = [cat_crossover(x, (idy*8), 32, 24) for idy in range(num_windows)]

#    windows.extend([(idx, idy) for idy in range(num_windows) if
#                    (crosses_per_window[idx][0] == 1) or
#                    (crosses_per_window[idx][0] == 0 and crosses_per_window[idx][1] > 0 and random.random() < 0.5) or
#                    (crosses_per_window[idx][0] == 0 and crosses_per_window[idx][1] == 0 and random.random() < 0.02)])



    total_windows += num_windows
    total_border += sum([ida == 0 and idb > 0 for ida, idb in crosses_per_window])
    total_pos += sum([ida == 1 for ida, idb in crosses_per_window])
    total_multi += sum([ida > 1 for ida, idb in crosses_per_window])

print(total_windows)
print(total_pos)
print(total_border)
print(total_multi)


#df = pd.DataFrame(windows, columns=["file_idx", "pos_idx"])

#df.to_csv(args.output, sep="\t", index=False)



