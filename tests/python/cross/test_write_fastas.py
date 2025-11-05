import pytest
from python.cross.write_fastas import write_fasta
import os
import pandas as pd

def test_write_fasta():
    chromosomes = ["chr1", "chr2"]
    fa_dir = "test_files"
    founder = 0

    fa_key = pd.DataFrame({"fa_chr": ["chr1", "chr1", "chr2", "chr2"],
                           "fa_start" : [0, 5, 0, 10],
                           "fa_end" : [5, 7, 10, 17],
                           "parent_chr" : ["chr1", "chr1", "chr2", "chr2"],
                           "parent_start" : [0, 3, 0, 10],
                           "parent_end" : [5, 5, 10, 17],
                           "parent": ["A", "B", "A", "B"]})

    fa_key.to_csv("0_key.bed", sep="\t", index=False, header=False)

    os.mkdir("recombinate_fastas")
    write_fasta(founder, chromosomes, fa_dir)

    expected_lines = [
        ">chr1",
        "AAAAA",
        "TT",
        ">chr2",
        "AAAAAAAAAA",
        "TTTTTTT"
    ]

    with open("recombinate_fastas/0.fa", "r") as f:
        lines = [line.strip() for line in f.readlines()]

    assert lines == expected_lines

    os.remove("0_key.bed")
    os.remove("test_files/A.fa.fai")
    os.remove("test_files/B.fa.fai")
    os.remove("recombinate_fastas/0.fa")
    os.removedirs("recombinate_fastas")