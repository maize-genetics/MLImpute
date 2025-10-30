import pandas as pd
import pysam
from multiprocessing import Pool, cpu_count
import argparse
import os


def write_fasta(founder, chromosomes, fa_dir):
    fasta_file = f"recombinate_fastas/{founder}.fa"  # create a new fasta file made up of parent fasta sequence
    with open(fasta_file, "w") as f:
        fasta_key = pd.read_csv(f"{founder}_key.bed", sep="\t", header=None, names=["fa_chr", "fa_start", "fa_end", "parent_chr", "parent_start", "parent_end", "parent"], index_col=False)
        for c in chromosomes:
            f.write(f">{c}\n")
        # print by chromosome
            for r, row in fasta_key[fasta_key["fa_chr"].astype(str) == str(c)].iterrows():
                with pysam.FastaFile(os.path.join(fa_dir, f"{row['parent']}.fa")) as fa:
                    seq = fa.fetch(str(row["parent_chr"]), int(row["parent_start"]), int(row["parent_end"]))
                f.write(f"{seq}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assembly-list", type=str, description="file containing full file paths and names for assembly fastas")
    parser.add_argument("--chromosome-list", type=str, help="file containing chromosomes names")
    parser.add_argument("--assembly-dir", type=str, help="directory containing assembly fasta files")
    args = parser.parse_args()

    os.makedirs("recombinate_fastas", exist_ok=True)

    assembly_founders = []

    with open(args.assembly_list) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                assembly_founders.append(parts[1])

    with open(args.chromosomes_list) as f:
        chromosomes = [line.strip() for line in f if line.strip()]

    for founder in range(len(assembly_founders)):
        # concurrently generate fasta files for each
        write_fasta(founder, chromosomes, args.assembly_dir)

    def run_founder(founder):
        """Wrapper for multiprocessing."""
        write_fasta(founder, chromosomes, args.assembly_dir)

    # Use all available cores, or limit manually (e.g. processes=8)
    with Pool(processes=cpu_count()) as pool:
        pool.map(run_founder, assembly_founders)