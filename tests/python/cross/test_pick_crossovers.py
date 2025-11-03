import pytest
from python.cross.pick_crossovers import simulate_rounds, convert_pop_to_key, shift_chrom_arm, merge_pop, Interval
import os
import pandas as pd


def test_simulate_rounds():
    chrom_lengths = {"chr1": 300_000_000, "chr2": 200_000_000}
    parents = ["A", "B", "C", "D"]

    pop = simulate_rounds(chrom_lengths, parents, rounds=1)

    assert len(pop) == 4

    for genome in pop:
        for chrom, length in chrom_lengths.items():
            assert genome[chrom][0].start == 0
            prev_end = genome[chrom][0].end
            for interval in genome[chrom][1:]:
                assert interval.start == prev_end
                prev_end = interval.end
            assert genome[chrom][-1].end == length

def test_convert_pop_to_key():
    pop = [{"chr1": [Interval(0, 100, "A"), Interval(100, 200, "B")],
            "chr2": [Interval(0, 300, "B"), Interval(300, 500, "A")]},
           {"chr1": [Interval(0, 100, "B"), Interval(100, 200, "A")],
            "chr2": [Interval(0, 300, "A"), Interval(300, 500, "B")]}]

    parents = ["A", "B"]

    convert_pop_to_key(pop, parents)

    A_df = pd.read_csv("A_refkey.bed", sep="\t", header=None, names=["chr", "start", "end", "founder"])
    B_df = pd.read_csv("B_refkey.bed", sep="\t", header=None, names=["chr", "start", "end", "founder"])

    A_df_expected = pd.DataFrame({"chr" : ["chr1", "chr1", "chr2", "chr2"],
                                  "start" : [0, 100, 0, 300],
                                  "end" : [100, 200, 300, 500],
                                  "founder" : [0, 1, 1, 0]})

    B_df_expected = pd.DataFrame({"chr" : ["chr1", "chr1", "chr2", "chr2"],
                                  "start" : [0, 100, 0, 300],
                                  "end" : [100, 200, 300, 500],
                                  "founder" : [1, 0, 0, 1]})

    pd.testing.assert_frame_equal(A_df, A_df_expected)
    pd.testing.assert_frame_equal(B_df, B_df_expected)

    os.remove("A_refkey.bed")
    os.remove("B_refkey.bed")

def test_shift_chrom_arm():
    pop = [{"chr1": [Interval(0, 100, "A"), Interval(100, 200, "B")],
            "chr2": [Interval(0, 300, "B"), Interval(300, 500, "A")]},
           {"chr1": [Interval(0, 100, "B"), Interval(100, 200, "A")],
            "chr2": [Interval(0, 300, "A"), Interval(300, 500, "B")]}]

    chrom = "chr1"

    pop_shifted_expected = [{"chr1": [Interval(200, 300, "A"), Interval(300, 400, "B")],
                             "chr2": [Interval(0, 300, "B"), Interval(300, 500, "A")]},
                            {"chr1": [Interval(200, 300, "B"), Interval(300, 400, "A")],
                             "chr2": [Interval(0, 300, "A"), Interval(300, 500, "B")]}]

    shift_chrom_arm(pop, chrom, length=200)

    assert pop == pop_shifted_expected

def test_merge_pop():

    pop1 = [{"chr1": [Interval(0, 100, "A"), Interval(100, 200, "B")],
             "chr2": [Interval(0, 300, "B"), Interval(300, 500, "A")]},
            {"chr1": [Interval(0, 100, "B"), Interval(100, 200, "A")],
             "chr2": [Interval(0, 300, "A"), Interval(300, 500, "B")]}]

    pop2 = [{"chr1": [Interval(200, 300, "A"), Interval(300, 400, "B")],
             "chr2": [Interval(500, 800, "B"), Interval(800, 1000, "A")]},
            {"chr1": [Interval(200, 300, "B"), Interval(300, 400, "A")],
             "chr2": [Interval(500, 800, "A"), Interval(800, 1000, "B")]}]

    merged_expected = [{"chr1": [Interval(0, 100, "A"), Interval(100, 200, "B"), Interval(200, 300, "A"), Interval(300, 400, "B")],
                        "chr2": [Interval(0, 300, "B"), Interval(300, 500, "A"), Interval(500, 800, "B"), Interval(800, 1000, "A")]},
                       {"chr1": [Interval(0, 100, "B"), Interval(100, 200, "A"), Interval(200, 300, "B"), Interval(300, 400, "A")],
                        "chr2": [Interval(0, 300, "A"), Interval(300, 500, "B"), Interval(500, 800, "A"), Interval(800, 1000, "B")]}]

    merged_actual = merge_pop(pop1, pop2)

    assert merged_actual == merged_expected