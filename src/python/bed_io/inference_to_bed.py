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
from bed_io.make_bed_contiguous import parse_bed, read_fai_lengths, make_contiguous, write_bed
import glob

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
    matrix_re = re.compile(r"(?P<sample>.+)_(?P<contig>(?:chr|chromosome)[^_]+)\.npy$", re.IGNORECASE)
    table_re = re.compile(r"(?P<sample>.+)\.(?:ps4g|_ps4g\.txt)$")
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
    for f in table_dir.glob("*ps4g"):
        m = table_re.match(f.name)
        if not m:
            continue
        sample = m.group("sample")
        samples[sample]["table"] = f
    return samples

def process_files(inference_dir, ps4g_dir, save_dir, bin_size=256):
    samples_with_matrices = associate_files_with_samples(inference_dir, ps4g_dir)

    for sample, info in samples_with_matrices.items():
        ps4g_file = info["table"]
        prediction_files = info["matrices"]

        if ps4g_file is None or not prediction_files:
            logging.info(f"Skipping incomplete sample {sample}")
            continue

        process_single_sample(prediction_files, ps4g_file, sample, save_dir, bin_size=bin_size)


def process_single_sample(prediction_files, ps4g_file, sample, save_dir, bin_size=256):
    # Build the index array for each founder sample
    index_array = build_index_lookup(ps4g_file)
    index_array.append(None)  # add extra index to represent "unlabelled" prediction
    df = pd.read_csv(ps4g_file, sep="\t", comment="#")
    df["refPosBinned"] *= bin_size  # Resolve the binned positions to actual positions
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
    final_predictions = np.load(prediction_files[contig], mmap_mode="r")
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
    parser.add_argument("--contiguous", action="store_true", help="Optional flag to make prediction path contiguous")
    parser.add_argument("--fai", default=None, help="Optional reference .fai to force chromosome end coordinates (if contiguous True)")
    parser.add_argument("--bin-size", type=int, default=256, help="bin size for positions")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    process_files(args.inference_dir, args.ps4g_dir, args.save_dir, bin_size=args.bin_size)
    chr_lengths = read_fai_lengths(args.fai) if args.fai else None

    if args.contiguous:
        bed_files = glob.glob(os.path.join(args.save_dir, "*.bed"))
        for file in bed_files:
            rows = parse_bed(file)
            out = make_contiguous(rows, chr_lengths=chr_lengths)
            new_file = file.replace(".bed", ".contiguous.bed")
            write_bed(out, new_file)

if __name__ == "__main__":
    main()