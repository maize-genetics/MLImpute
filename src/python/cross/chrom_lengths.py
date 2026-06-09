import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
returns a dictionary mapping chromosomes to chromosome length for a fasta file
"""
def chrom_lengths(fasta_file, exclude_scaffolds=True):
    fai = fasta_file + ".fai"
    if not os.path.exists(fai):
        # build the index once (very fast; re-used later)
        subprocess.run(["samtools", "faidx", fasta_file], check=True)

    chrom_dict = {}
    with open(fai) as f:
        for line in f:
            name, length, *_ = line.rstrip("\n").split("\t")
            if exclude_scaffolds:
                if name.startswith(("chr", "chromosome", "Chr", "CHR")):
                    chrom_dict[name] = int(length)
            else:
                chrom_dict[name] = int(length)
    return chrom_dict

"""
returns a dictionary mapping assemblies to a dictionary mapping chromosome to chromosome length
"""
def chrom_lengths_dicts(assembly_list, exclude_scaffolds=True, max_workers=8):
    chrom_dicts = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(chrom_lengths, asm, exclude_scaffolds): asm
            for asm in assembly_list
        }
        for fut in as_completed(futs):
            asm = futs[fut]
            key = asm.split("/")[-1].split(".fa")[0]
            chrom_dicts[key] = fut.result()
    return chrom_dicts