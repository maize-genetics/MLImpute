#!/usr/bin/env python
"""Runnable CLI: impute the best haploid/diploid haplotype path(s) from a PS4G file.

PyTorch port of phg_v2's `ImputePathFromPs4g` CLI (net.maizegenetics.phgv2.pathing.ropebwt),
using python.hmm.ps4g_hmm for emissions/transitions built on the fly from the PS4G file and
python.hmm.viterbi.viterbi_decode (CPU or GPU) for path finding.

Usage:
    python src/python/hmm/impute_ps4g.py --read-file sample.ps4g --out-path-dir out/
    python src/python/hmm/impute_ps4g.py --path-keyfile keyfile.txt --out-path-dir out/ \\
        --path-type diploid --n-parents 4 --inbreed-coef 0.5
"""

import argparse
import sys
from pathlib import Path
from typing import List, NamedTuple, Sequence, Tuple

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from python.hmm.ps4g_hmm import (  # noqa: E402
    build_contig_readmap,
    find_diploid_path,
    find_haploid_path,
    most_likely_parents,
    parse_gamete_index_map,
    resolve_device,
)


class KeyFileData(NamedTuple):
    sample_name: str
    ps4g_file: str


def read_keyfile(path: str) -> List[KeyFileData]:
    with open(path) as fh:
        lines = [line.rstrip("\n") for line in fh if line.strip() != ""]
    if not lines:
        raise ValueError(f"Key file {path} is empty.")
    header = lines[0].split("\t")
    if "sampleName" not in header or "filename" not in header:
        raise ValueError(f"Key file {path} must have columns named sampleName and filename.")
    sample_idx = header.index("sampleName")
    file_idx = header.index("filename")
    return [
        KeyFileData(cols[sample_idx], cols[file_idx])
        for cols in (line.split("\t") for line in lines[1:])
    ]


def read_single_file(path: str) -> List[KeyFileData]:
    stem = Path(path).name
    for suffix in (".ps4g", ".txt", ".gz"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return [KeyFileData(stem, path)]


def _write_midpoint_rows(writer, contig: str, path: Sequence[Tuple], bin_size: int) -> None:
    """Write BED rows using the midpoint-coordinate scheme from ImputePathFromPs4g.kt:
    each row's boundary sits halfway (in bin units) between its position and its
    neighbor's, converted to bp via bin_size.
    """
    positions = [row[0] for row in path]
    length = len(path)

    if length == 1:
        writer.write("\t".join([contig, "1", str(positions[0] * bin_size), *path[0][1:]]) + "\n")
        return

    def midpoint(i: int, j: int) -> int:
        return (positions[i] + positions[j]) // 2 * bin_size

    start = 1
    end = midpoint(0, 1)
    writer.write("\t".join([contig, str(start), str(end), *path[0][1:]]) + "\n")

    for k in range(1, length - 1):
        start = midpoint(k - 1, k) + 1
        end = midpoint(k, k + 1)
        writer.write("\t".join([contig, str(start), str(end), *path[k][1:]]) + "\n")

    k = length - 1
    start = midpoint(k - 1, k) + 1
    end = positions[k] * bin_size
    writer.write("\t".join([contig, str(start), str(end), *path[k][1:]]) + "\n")


def _write_path_file(output_path: Path, header: str, contig_paths: Sequence[Tuple[str, Sequence[Tuple]]], bin_size: int) -> None:
    with open(output_path, "w") as writer:
        writer.write(header + "\n")
        for contig, path in contig_paths:
            if path:
                _write_midpoint_rows(writer, contig, path, bin_size)


def impute_haploid_path(entries: Sequence[KeyFileData], out_dir: Path, prob_correct: float, prob_same: float, bin_size: int, device) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        gamete_index_map = parse_gamete_index_map(entry.ps4g_file)
        contig_readmap = build_contig_readmap(entry.ps4g_file)

        contig_paths = []
        for contig in sorted(contig_readmap.keys()):
            path = find_haploid_path(
                contig, gamete_index_map, contig_readmap[contig], prob_correct, prob_same, device
            )
            contig_paths.append((contig, path))

        output_path = out_dir / f"{entry.sample_name}_imputed_path.bed"
        _write_path_file(output_path, "chrom\tstart\tend\tparent1", contig_paths, bin_size)


def impute_diploid_path(
    entries: Sequence[KeyFileData],
    out_dir: Path,
    prob_correct: float,
    prob_same: float,
    inbreed_coef: float,
    n_parents: int,
    bin_size: int,
    device,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        gamete_index_map = parse_gamete_index_map(entry.ps4g_file)
        contig_readmap = build_contig_readmap(entry.ps4g_file)
        # Matches ImputePathFromPs4g.imputeDiploidPath, which skips scaffold contigs.
        contigs = sorted(c for c in contig_readmap.keys() if not c.startswith("scaf"))

        n_genomes = len(gamete_index_map)
        if 1 <= n_parents < n_genomes:
            parent_set = most_likely_parents(contig_readmap, contigs, n_parents)
        else:
            parent_set = set(gamete_index_map.keys())

        contig_paths = []
        for contig in contigs:
            path = find_diploid_path(
                contig,
                gamete_index_map,
                contig_readmap[contig],
                parent_set,
                prob_correct,
                prob_same,
                inbreed_coef,
                device,
            )
            contig_paths.append((contig, path))

        output_path = out_dir / f"{entry.sample_name}_imputed_path.txt"
        _write_path_file(output_path, "chrom\tstart\tend\tparent1\tparent2", contig_paths, bin_size)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Impute best haplotype path(s) from a PS4G file using a PyTorch Viterbi HMM.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--path-keyfile", help="Tab-delimited key file with sampleName and filename columns pointing to PS4G files.")
    input_group.add_argument("--read-file", help="Path to a single PS4G file.")
    parser.add_argument("--out-path-dir", required=True, help="Directory to write imputed path file(s) to.")
    parser.add_argument("--path-type", choices=["haploid", "diploid"], default="haploid")
    parser.add_argument("--prob-correct", type=float, default=0.98, help="Probability a read maps to the correct haplotype.")
    parser.add_argument("--prob-same", type=float, default=0.9999, help="Probability of staying on the same gamete between adjacent positions.")
    parser.add_argument("--inbreed-coef", type=float, default=0.0, help="Inbreeding coefficient in [0,1], diploid only.")
    parser.add_argument("--n-parents", type=int, default=0, help="Restrict diploid imputation to this many likely parents (0 = use all).")
    parser.add_argument("--bin-size", type=int, default=256, help="Bin size used to create the PS4G file.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'.")
    return parser


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    entries = read_keyfile(args.path_keyfile) if args.path_keyfile else read_single_file(args.read_file)
    device = resolve_device(args.device)
    out_dir = Path(args.out_path_dir)

    if args.path_type == "haploid":
        impute_haploid_path(entries, out_dir, args.prob_correct, args.prob_same, args.bin_size, device)
    else:
        impute_diploid_path(
            entries, out_dir, args.prob_correct, args.prob_same, args.inbreed_coef, args.n_parents, args.bin_size, device
        )


if __name__ == "__main__":
    main()
