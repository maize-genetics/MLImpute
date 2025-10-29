import pandas as pd
import pysam
from multiprocessing import Pool, cpu_count


def write_fasta(founder, chromosomes):
    fasta_file = f"{founder}.fa"  # create a new fasta file made up of parent fasta sequence
    with open(fasta_file, "w") as f:
        fasta_key = pd.read_csv(f"{founder}_key.bed", sep="\t", header=None, names=["fa_chr", "fa_start", "fa_end", "parent_chr", "parent_start", "parent_end", "parent"], index_col=False)
        for c in chromosomes:
            f.write(f">{c}\n")
        # print by chromosome
            for r, row in fasta_key[fasta_key["fa_chr"].astype(str) == str(c)].iterrows():
                with pysam.FastaFile(f"/workdir/smm477/uncrossed_phg/updated_fastas/{row["parent"]}.fa") as fa:
                    seq = fa.fetch(str(row["parent_chr"]), int(row["parent_start"]), int(row["parent_end"]))
                f.write(f"{seq}\n")

if __name__ == "__main__":
    NAM_founders = ["CML228", "CML322", "CML69", "Ki11", "M162W", "Ms71", "Oh43", "B97", "CML247", "CML333", "HP301", "Ki3",
                    "M37W", "NC350", "Oh7B", "Tzi8", "CML103", "CML277", "CML52", "Il14H", "Ky21", "Mo18W", "NC358", "P39"]

    chromosomes = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10"]

    for founder in range(len(NAM_founders)):
        # concurrently generate fasta files for each
        write_fasta(founder, chromosomes)

    def run_founder(founder):
        """Wrapper for multiprocessing."""
        write_fasta(founder, chromosomes)

    # Use all available cores, or limit manually (e.g. processes=8)
    with Pool(processes=cpu_count()) as pool:
        pool.map(run_founder, NAM_founders)