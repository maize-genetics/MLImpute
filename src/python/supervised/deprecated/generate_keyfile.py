# script to generate keyfile for use with labeled datasets
# most datasets only need the file name and path,
# but additional normalization parameters are calculated for certain vision models

import numpy as np
import pandas as pd
import argparse
import os
import itertools

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-parents", type=int, default=24, help="number of parents")
    parser.add_argument("--dir", type=str, required=True, help="directory containing the npy files")
    parser.add_argument("--output", "-o", type=str, required=True, help="output keyfile")

    args = parser.parse_args()
    return args

args = parse_args()

file_names = []
for file in os.listdir(args.dir):
    if file.endswith(".npy"):
        file_names.append(file)

# group by dataset
groups = itertools.groupby(file_names, lambda x: x.split("chr")[0])

chrom_mean = {}
chrom_sd = {}
global_mean = {}
global_sd = {}
chrom_len = {}

for k,g in groups:
    x = None
    files = []
    for file in g:
        files.append(file)
        y = np.load(args.dir + "/" + file)

        chrom_mean[file] = np.mean(y[:, 0:args.num_parents])
        chrom_sd[file] = np.std(y[:, 0:args.num_parents])
        chrom_len[file] = y.shape[0]

        if x is not None:
            x = np.concatenate((x, y[:, 0:args.num_parents]), axis=0)
        else:
            x = y[:, 0:args.num_parents]

    g_mean = np.mean(x)
    g_sd = np.mean(y)

    for file in files:
        global_mean[file] = g_mean
        global_sd[file] = g_sd

tups = [(name, args.dir + "/" + name, chrom_len[name], chrom_mean[name], chrom_sd[name], global_mean[name], global_sd[name])
        for name in file_names]

df = pd.DataFrame(tups, columns=["file_name", "path", "length", "chrom_mean", "chrom_stdev", "global_mean", "global_stdev"])

df.to_csv(args.output, sep="\t", index=False)






