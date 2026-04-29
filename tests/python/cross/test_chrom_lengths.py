import pytest
from python.cross.chrom_lengths import chrom_lengths, chrom_lengths_dicts
import os

def test_chrom_lengths():
    fasta_file_A = "test_files/A.fa"
    fasta_file_B = "test_files/B.fa"

    chrom_dict_A = chrom_lengths(fasta_file_A, exclude_scaffolds=True)
    chrom_dict_B = chrom_lengths(fasta_file_B, exclude_scaffolds=True)

    assert chrom_dict_A == {"chr1": 10, "chr2": 20}
    assert chrom_dict_B == {"chr1": 8, "chr2": 18}

    chrom_dict_A_scaf = chrom_lengths(fasta_file_A, exclude_scaffolds=False)
    chrom_dict_B_scaf = chrom_lengths(fasta_file_B, exclude_scaffolds=False)

    assert chrom_dict_A_scaf == {"chr1": 10, "chr2": 20, "scaf1": 10}
    assert chrom_dict_B_scaf == {"chr1": 8, "chr2": 18, "scaf1": 10}

    os.remove("test_files/A.fa.fai")
    os.remove("test_files/B.fa.fai")

def test_chrom_lengths_dict():
    assembly_list = ["test_files/A.fa", "test_files/B.fa"]

    chrom_dicts = chrom_lengths_dicts(assembly_list, exclude_scaffolds=True)
    print(chrom_dicts)

    assert chrom_dicts == {"A": {"chr1": 10, "chr2": 20}, "B": {"chr1": 8, "chr2": 18}}
    os.remove("test_files/A.fa.fai")
    os.remove("test_files/B.fa.fai")