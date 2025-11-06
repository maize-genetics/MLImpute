import numpy as np

out_file_name = "foo.npy"
label_file_name = "bar.npy"

num_parents = 24
length = 10_000_000

min_mean = 1
max_mean = 100

min_sd = 1
max_sd = 50

min_dist = 100
max_dist = 5000

out_array = np.zeros((length, num_parents+1), dtype=int)
crossovers = []

current_crossover = 0
previous_hap = -1

while current_crossover < length:
    next_crossover = current_crossover + np.random.randint(min_dist, max_dist)
    print(next_crossover)

    if next_crossover > length:
        next_crossover = length

    crossovers.append(next_crossover)

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
np.save(label_file_name, np.array(crossovers, dtype=int))

