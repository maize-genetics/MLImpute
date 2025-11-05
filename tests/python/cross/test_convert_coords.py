import pytest
from python.cross.convert_coords import build_fasta_keys, adjust_coords
import os
import pandas as pd

def test_build_fasta_keys():
    parents = ["A", "B"]

    A_key_df = pd.DataFrame({"parent_chr" : ["chr1", "chr1"],
                             "parent_start" : [0, 5],
                             "parent_end" : [5, 10],
                             "ref_chr" : ["chr1", "chr1"],
                             "ref_start" : [0, 4],
                             "ref_end" : [4, 8],
                             "founder" : [0, 1]})

    B_key_df = pd.DataFrame({"parent_chr" : ["chr1", "chr1"],
                             "parent_start" : [0, 3],
                             "parent_end" : [3, 10],
                             "ref_chr" : ["chr1", "chr1"],
                             "ref_start" : [0, 4],
                             "ref_end" : [4, 8],
                             "founder" : [1, 0]})

    A_key_df.to_csv('A_key.bed', sep='\t', header=False)
    B_key_df.to_csv('B_key.bed', sep='\t', header=False)

    build_fasta_keys(parents, 0)
    build_fasta_keys(parents, 1)

    founder0_key_actual = pd.read_csv('0_key.bed', sep='\t', header=None, names=["fa_chr", "fa_start", "fa_end", "parent_chr", "parent_start", "parent_end", "parent"])
    founder1_key_actual = pd.read_csv('1_key.bed', sep='\t', header=None, names=["fa_chr", "fa_start", "fa_end", "parent_chr", "parent_start", "parent_end", "parent"])

    founder0_key_expected = pd.DataFrame({"fa_chr" : ["chr1", "chr1"],
                                          "fa_start" : [0, 5],
                                          "fa_end" : [5, 12],
                                          "parent_chr" : ["chr1", "chr1"],
                                          "parent_start" : [0, 3],
                                          "parent_end" : [5, 10],
                                          "parent" : ["A", "B"]})

    founder1_key_expected = pd.DataFrame({"fa_chr" : ["chr1", "chr1"],
                                          "fa_start" : [0, 3],
                                          "fa_end" : [3, 8],
                                          "parent_chr" : ["chr1", "chr1"],
                                          "parent_start" : [0, 5],
                                          "parent_end" : [3, 10],
                                          "parent" : ["B", "A"]})

    pd.testing.assert_frame_equal(founder0_key_actual, founder0_key_expected)
    pd.testing.assert_frame_equal(founder1_key_actual, founder1_key_expected)

    os.remove("A_key.bed")
    os.remove("B_key.bed")

    os.remove("0_key.bed")
    os.remove("1_key.bed")

def test_adjust_coords():
    df = pd.DataFrame({"parent_chr": ["chr1", "chr1", "chr1", "chr1"],
                       "parent_start": [10, 250, 330, 417],
                       "parent_end": [225, 312, 415, 534],
                       "parent_founder": [0, 1, 2, 1]})
    length = 550

    adjusted_df_expected = pd.DataFrame({"parent_chr": ["chr1", "chr1", "chr1", "chr1"],
                                         "parent_start": [0, 225, 312, 415],
                                         "parent_end": [225, 312, 415, 550],
                                         "parent_founder": [0, 1, 2, 1]})

    adjusted_df_actual = adjust_coords(df, length)

    pd.testing.assert_frame_equal(adjusted_df_expected, adjusted_df_actual)