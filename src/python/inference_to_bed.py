import argparse
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import os
from ps4g_io.ps4g import build_index_lookup
from bed_io.bed import output_bed_file
import logging


def write_bed_for_single_contig(
    output_bed,
    contig,
    positions,
    final_predictions,
    index_array,
    collapse_bed_regions=True,
    ploidy=2,
):
    positions = np.asarray(positions, dtype=np.int64)
    chroms = np.full(len(positions), contig, dtype=object)

    final_predictions = np.asarray(final_predictions)

    # ---- diploid normalization ----
    if final_predictions.ndim == 1:
        if ploidy != 1:
            final_predictions = np.repeat(final_predictions[:, None], ploidy, axis=1)
    elif final_predictions.ndim == 2:
        if final_predictions.shape[1] != ploidy:
            raise ValueError("Prediction ploidy mismatch")

    # ---- truncate to prediction length ----
    N = min(len(positions), len(final_predictions))

    positions = positions[:N]
    chroms = chroms[:N]
    final_predictions = final_predictions[:N]

    output_bed_file(
        output_bed=output_bed,
        chroms=chroms,
        final_predictions=final_predictions,
        index_array=index_array,
        positions=positions,
        collapse_bed_regions=collapse_bed_regions,
    )

def associate_files_with_samples(inference_dir, ps4g_dir):
    # some regex patterns found from chatgpt  this allows us to parse out the sample name and the contig name
    matrix_re = re.compile(r"(?P<sample>.+)_(?P<contig>chr[^_]+)\.npy")
    table_re = re.compile(r"(?P<sample>.+)_ps4g\.txt")
    matrix_dir = Path(inference_dir)
    table_dir = Path(ps4g_dir)
    samples = defaultdict(lambda: {"matrices": {}, "table": None})
    for f in matrix_dir.glob("*.npy"):
        m = matrix_re.match(f.name)
        if not m:
            continue
        sample = m.group("sample")
        contig = m.group("contig")
        samples[sample]["matrices"][contig] = f
    for f in table_dir.glob("*_ps4g.txt"):
        m = table_re.match(f.name)
        if not m:
            continue
        sample = m.group("sample")
        samples[sample]["table"] = f
    return samples

def process_files(inference_dir, ps4g_dir, save_dir):
    samples_with_matrices = associate_files_with_samples(inference_dir, ps4g_dir)

    for sample, info in samples_with_matrices.items():
        ps4g_file = info["table"]
        prediction_files = info["matrices"]

        if ps4g_file is None or not prediction_files:
            logging.info(f"Skipping incomplete sample {sample}")
            continue

        process_single_sample(prediction_files, ps4g_file, sample, save_dir)


def process_single_sample(prediction_files, ps4g_file, sample, save_dir):
    # Build the index array for each founder sample
    index_array = build_index_lookup(ps4g_file)
    index_array.append(None)  # add extra index to represent "unlabelled" prediction
    df = pd.read_csv(ps4g_file, sep="\t", comment="#")
    df["refPosBinned"] *= 256  # Resolve the binned positions to actual positions
    contig_positions = (
        df.groupby("refContig")["refPosBinned"]
        .apply(lambda x: sorted(set(x)))
        .to_dict()
    )

    for contig, positions in contig_positions.items():
        if contig not in prediction_files:
            continue  # or warn

        process_contig(contig, index_array, positions, prediction_files, sample, save_dir)


def process_contig(contig, index_array, positions, prediction_files, sample, save_dir):
    final_predictions = np.load(prediction_files[contig], mmap_mode="r").flatten()
    write_bed_for_single_contig(
        output_bed=os.path.join(save_dir, f"{sample}_{contig}_imputed.bed"),
        contig=contig,
        positions=positions,
        final_predictions=final_predictions,
        index_array=index_array,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--inference-dir","-i", type=str, required=True, help="path to the inferred numpy data")
    parser.add_argument("--ps4g-dir", "-p", type=str, required=True, help="path to the input PS4G data")
    parser.add_argument("--save-dir","-s", type=str, required=True, help="path to the output bed files")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    process_files(args.inference_dir, args.ps4g_dir, args.save_dir)

if __name__ == "__main__":
    main()