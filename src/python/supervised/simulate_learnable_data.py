# Temporary file, simulate a dataset that has a basic pattern to learn
# using this so that I can troubleshoot causal language issues without
# having the actual traning datasets

import numpy as np
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--length", "-l", default=10_000_000, type=int, help="contig length")
parser.add_argument("--out", "-o", default="foo.npy", help="output file name")

args = parser.parse_args()

out_file_name = args.out

num_parents = 24
length = args.length

min_mean = 1
max_mean = 100

min_sd = 1
max_sd = 50

min_dist = 100
max_dist = 5000

out_array = np.zeros((length, num_parents+1), dtype=int)

current_crossover = 0
previous_hap = -1

while current_crossover < length:
    next_crossover = current_crossover + np.random.randint(min_dist, max_dist)

    if next_crossover > length:
        next_crossover = length

    haplotype = np.random.randint(0, num_parents)

    while haplotype == previous_hap:
        haplotype = np.random.randint(0, num_parents)

    out_array[current_crossover:next_crossover, num_parents] = haplotype

    for idx in range(num_parents):
        mean = np.random.randint(min_mean, max_mean)
        sd = np.random.randint(min_sd, max_sd)

        dist = np.random.normal(mean, sd, (next_crossover-current_crossover))

        out_array[current_crossover:next_crossover, idx] = dist

    current_crossover = next_crossover
    previous_hap = haplotype

np.save(out_file_name, out_array)
