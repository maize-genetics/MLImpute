import unittest
from typing import List, Dict, Tuple, Union
from create_founders import Interval, Mosaic, Genome, Line
from create_founders import pick_crossovers, make_founder_genome, _intervals_covering, recombine_two_mosaics, cross_lines
from create_founders import simulate_rounds, mean_segment_size, founder_contributions
from create_founders import create_df, convert_pop_to_key, shift_chrom_arm, merge_pop
from create_founders import convert_coord

class TestPopSetUp(unittest.TestCase):
    def setUp(self):
        self.chrom_lengths = {"chr1" : 30_000_000, "chr2" : 15_000_000, "scaf_1" : 1_000_000}
        self.founders = ["B97", "CML69", "Tzi8", "Mo37W"]

    def test_make_founder_genome(self):
        initial_pop : List[Line] = [make_founder_genome(self.chrom_lengths, f) for f in self.founders]
        self.assertEqual(4, len(initial_pop))

        self.assertEqual(3, len(initial_pop[0]))
        self.assertEqual(3, len(initial_pop[1]))
        self.assertEqual(3, len(initial_pop[2]))
        self.assertEqual(3, len(initial_pop[3]))

        self.assertEqual(1, len(initial_pop[0]["chr1"]))
        self.assertEqual(1, len(initial_pop[1]["chr1"]))
        self.assertEqual(1, len(initial_pop[2]["chr1"]))
        self.assertEqual(1, len(initial_pop[3]["chr1"]))

        self.assertEqual(1, len(initial_pop[0]["chr2"]))
        self.assertEqual(1, len(initial_pop[1]["chr2"]))
        self.assertEqual(1, len(initial_pop[2]["chr2"]))
        self.assertEqual(1, len(initial_pop[3]["chr2"]))

        self.assertEqual(1, len(initial_pop[0]["scaf_1"]))
        self.assertEqual(1, len(initial_pop[1]["scaf_1"]))
        self.assertEqual(1, len(initial_pop[2]["scaf_1"]))
        self.assertEqual(1, len(initial_pop[3]["scaf_1"]))

        self.assertIn("chr1", initial_pop[0])
        self.assertIn("chr1", initial_pop[1])
        self.assertIn("chr1", initial_pop[2])
        self.assertIn("chr1", initial_pop[3])

        self.assertIn("chr2", initial_pop[0])
        self.assertIn("chr2", initial_pop[1])
        self.assertIn("chr2", initial_pop[2])
        self.assertIn("chr2", initial_pop[3])

        self.assertIn("scaf_1", initial_pop[0])
        self.assertIn("scaf_1", initial_pop[1])
        self.assertIn("scaf_1", initial_pop[2])
        self.assertIn("scaf_1", initial_pop[3])

        self.assertEqual(initial_pop[0]["chr1"][0].start, 0)
        self.assertEqual(initial_pop[1]["chr1"][0].start, 0)
        self.assertEqual(initial_pop[2]["chr1"][0].start, 0)
        self.assertEqual(initial_pop[3]["chr1"][0].start, 0)

        self.assertEqual(initial_pop[0]["chr1"][0].end, 30_000_000)
        self.assertEqual(initial_pop[1]["chr1"][0].end, 30_000_000)
        self.assertEqual(initial_pop[2]["chr1"][0].end, 30_000_000)
        self.assertEqual(initial_pop[3]["chr1"][0].end, 30_000_000)

        self.assertEqual(initial_pop[0]["chr1"][0].founder, "B97")
        self.assertEqual(initial_pop[1]["chr1"][0].founder, "CML69")
        self.assertEqual(initial_pop[2]["chr1"][0].founder, "Tzi8")
        self.assertEqual(initial_pop[3]["chr1"][0].founder, "Mo37W")

        self.assertEqual(initial_pop[0]["chr2"][0].start, 0)
        self.assertEqual(initial_pop[1]["chr2"][0].start, 0)
        self.assertEqual(initial_pop[2]["chr2"][0].start, 0)
        self.assertEqual(initial_pop[3]["chr2"][0].start, 0)

        self.assertEqual(initial_pop[0]["chr2"][0].end, 15_000_000)
        self.assertEqual(initial_pop[1]["chr2"][0].end, 15_000_000)
        self.assertEqual(initial_pop[2]["chr2"][0].end, 15_000_000)
        self.assertEqual(initial_pop[3]["chr2"][0].end, 15_000_000)

        self.assertEqual(initial_pop[0]["chr2"][0].founder, "B97")
        self.assertEqual(initial_pop[1]["chr2"][0].founder, "CML69")
        self.assertEqual(initial_pop[2]["chr2"][0].founder, "Tzi8")
        self.assertEqual(initial_pop[3]["chr2"][0].founder, "Mo37W")

        self.assertEqual(initial_pop[0]["scaf_1"][0].start, 0)
        self.assertEqual(initial_pop[1]["scaf_1"][0].start, 0)
        self.assertEqual(initial_pop[2]["scaf_1"][0].start, 0)
        self.assertEqual(initial_pop[3]["scaf_1"][0].start, 0)

        self.assertEqual(initial_pop[0]["scaf_1"][0].end, 1_000_000)
        self.assertEqual(initial_pop[1]["scaf_1"][0].end, 1_000_000)
        self.assertEqual(initial_pop[2]["scaf_1"][0].end, 1_000_000)
        self.assertEqual(initial_pop[3]["scaf_1"][0].end, 1_000_000)

        self.assertEqual(initial_pop[0]["scaf_1"][0].founder, "B97")
        self.assertEqual(initial_pop[1]["scaf_1"][0].founder, "CML69")
        self.assertEqual(initial_pop[2]["scaf_1"][0].founder, "Tzi8")
        self.assertEqual(initial_pop[3]["scaf_1"][0].founder, "Mo37W")

class TestCross(unittest.TestCase):
    def setUp(self):
        self.chrom_lengths = {"chr1" : 30_000_000, "chr2" : 15_000_000, "scaf_1" : 1_000_000}
        self.founders = ["B97", "CML69", "Tzi8", "Mo37W"]
        self.initial_pop : List[Line] = [make_founder_genome(self.chrom_lengths, f) for f in self.founders]

    def test_no_empty_intervals(self):
        for i, m in enumerate(self.initial_pop):
            for c in m:
                for interval in m[c]:
                    self.assertLess(interval.start, interval.end)

class TestConvert(unittest.TestCase):
    def setUp(self):
        self.s_info = {
            "chr": "chr1",
            "start": 62,
            "end": 62+36,
            "length": 36,
            "strand": "+",
            "chr_length": 600,
            "seq": "TCACGGCCCTGTGCT---CACTCCTGAACGCTCCGT--CTA"
        }

        self.neg_s_info = {
            "chr": "chr1",
            "start": 502,
            "end": 502+36,
            "length": 36,
            "strand": "-",
            "chr_length": 600,
            "seq": "TCACGGCCCTGTGCT---CACTCCTGAACGCTCCGT--CTA"
        }

        self.ref_info = {
            "chr": "chr1",
            "start": 57,
            "end": 57+38,
            "length": 38,
            "strand": "+",
            "chr_length": 500,
            "seq": "TCACCCCCCTGTGCTCAACACTCCTG--CGCTCCGTTGC-A"
        }

        self.s_info2 = {
            "chr": "chr1",
            "start": 62,
            "end": 62+36,
            "length": 36,
            "strand": "+",
            "chr_length": 600,
            "seq": "TCACGGCCCTGTGCTCACTCCTGAACGCTCCGTCTA"
        }

        self.neg_s_info2 = {
            "chr": "chr1",
            "start": 502,
            "end": 502+36,
            "length": 36,
            "strand": "-",
            "chr_length": 600,
            "seq": "TCACGGCCCTGTGCTCACTCCTGAACGCTCCGTCTA"
        }

        self.ref_info2 = {
            "chr": "chr1",
            "start": 57,
            "end": 57+36,
            "length": 36,
            "strand": "+",
            "chr_length": 500,
            "seq": "TCACGGCCCTGTGCTCACTCCTGAACGCTCCGTCTA"
        }

        self.s_info3 = {
            "chr": "chr1",
            "start": 62,
            "end": 62+35,
            "length": 35,
            "strand": "+",
            "chr_length": 600,
            "seq": "TCAAA-GGGCCCTGTGCTCAC--CTGAACGC-CCGTCTA"
        }

        self.neg_s_info3 = {
            "chr": "chr1",
            "start": 503,
            "end": 503+35,
            "length": 35,
            "strand": "-",
            "chr_length": 600,
            "seq": "TCAAA-GGGCCCTGTGCTCAC--CTGAACGC-CCGTCTA"
        }

        self.ref_info3 = {
            "chr": "chr1",
            "start": 57,
            "end": 57+30,
            "length": 30,
            "strand": "+",
            "chr_length": 500,
            "seq": "TC--ATG-GCCCTGT---CACTCCTGAACGCTCC---TA"
        }

        self.s_info4 = {
            "chr": "chr1",
            "start": 62,
            "end": 62+26,
            "length": 26,
            "strand": "+",
            "chr_length": 600,
            "seq": "TCAAA---------GCTCACCTGAACGCCCGTCTA"
        }

        self.neg_s_info4 = {
            "chr": "chr1",
            "start": 512,
            "end": 512+26,
            "length": 26,
            "strand": "-",
            "chr_length": 600,
            "seq": "TCAAA---------GCTCACCTGAACGCCCGTCTA"
        }

        self.ref_info4 = {
            "chr": "chr1",
            "start": 57,
            "end": 57+35,
            "length": 35,
            "strand": "+",
            "chr_length": 500,
            "seq": "TCAAAGGGCCCTGTGCTCACCTGAACGCCCGTCTA"
        }

    def test_overlap_right_no_gaps(self): # Case 1.1
        parent_start, parent_end = convert_coord(self.s_info, self.ref_info, ref_start=40, ref_end=69)
        self.assertEqual(parent_start, 62)
        self.assertEqual(parent_end, 74)

    def test_spanning_no_gaps(self): # Case 2.1
        parent_start, parent_end = convert_coord(self.s_info2, self.ref_info2, ref_start=0, ref_end=120)
        self.assertEqual(parent_start, 62)
        self.assertEqual(parent_end, 98)

    def test_fully_contained_no_gaps(self): # Case 3.1
        parent_start, parent_end = convert_coord(self.s_info, self.ref_info, ref_start=58, ref_end=69)
        self.assertEqual(parent_start, 63)
        self.assertEqual(parent_end, 74)

    def test_overlap_left_no_gaps(self): # Case 4.1
        parent_start, parent_end = convert_coord(self.s_info, self.ref_info, ref_start=94, ref_end=100)
        self.assertEqual(parent_start, 97)
        self.assertEqual(parent_end, 98)

    def test_overlap_right_with_gaps(self): # Case 1.2
        parent_start, parent_end = convert_coord(self.s_info3, self.ref_info3, ref_start=0, ref_end=66)
        self.assertEqual(parent_start, 62)
        self.assertEqual(parent_end, 73)

    def test_spanning_with_gaps(self): # Case 2.2
        parent_start, parent_end = convert_coord(self.s_info3, self.ref_info3, ref_start=0, ref_end=200)
        self.assertEqual(parent_start, 62)
        self.assertEqual(parent_end, 97)

    def test_fully_contained_with_gaps(self): # Case 3.2
        parent_start, parent_end = convert_coord(self.s_info3, self.ref_info3, ref_start=66, ref_end=79)
        self.assertEqual(parent_start, 73)
        self.assertEqual(parent_end, 87)

    def test_overlap_left_with_gaps(self): # Case 4.2
        parent_start, parent_end = convert_coord(self.s_info3, self.ref_info3, ref_start=80, ref_end=200)
        self.assertEqual(parent_start, 88)
        self.assertEqual(parent_end, 97)

    def test_fully_contained_start_gap(self): # Case 3.3
        parent_start, parent_end = convert_coord(self.s_info3, self.ref_info3, ref_start=60, ref_end=79)
        self.assertEqual(parent_start, 67)
        self.assertEqual(parent_end, 87)

    def test_overlap_left_start_gap(self): # Case 4.3
        parent_start, parent_end = convert_coord(self.s_info3, self.ref_info3, ref_start=72, ref_end=200)
        self.assertEqual(parent_start, 82)
        self.assertEqual(parent_end, 97)

    def test_overlap_right_end_gap(self): # Case 1.3
        parent_start, parent_end = convert_coord(self.s_info3, self.ref_info3, ref_start=0, ref_end=73)
        self.assertEqual(parent_start, 62)
        self.assertEqual(parent_end, 82)

    def test_fully_contained_end_gap(self): # Case 3.4
        parent_start, parent_end = convert_coord(self.s_info3, self.ref_info3, ref_start=66, ref_end=83)
        self.assertEqual(parent_start, 73)
        self.assertEqual(parent_end, 90)

    def test_fully_contained_empty(self): # Case 3.5
        parent_start, parent_end = convert_coord(self.s_info4, self.ref_info4, ref_start=62, ref_end=70)
        self.assertEqual(parent_start, None)
        self.assertEqual(parent_end, None)

    def test_neg_overlap_right_no_gaps(self): # Case 1.1
        parent_start, parent_end = convert_coord(self.neg_s_info, self.ref_info, ref_start=40, ref_end=69)
        self.assertEqual(parent_start, 62)
        self.assertEqual(parent_end, 74)

    def test_neg_spanning_no_gaps(self): # Case 2.1
        parent_start, parent_end = convert_coord(self.neg_s_info2, self.ref_info2, ref_start=0, ref_end=120)
        self.assertEqual(parent_start, 62)
        self.assertEqual(parent_end, 98)

    def test_neg_fully_contained_no_gaps(self): # Case 3.1
        parent_start, parent_end = convert_coord(self.neg_s_info, self.ref_info, ref_start=58, ref_end=69)
        self.assertEqual(parent_start, 63)
        self.assertEqual(parent_end, 74)

    def test_neg_overlap_left_no_gaps(self): # Case 4.1
        parent_start, parent_end = convert_coord(self.neg_s_info, self.ref_info, ref_start=94, ref_end=100)
        self.assertEqual(parent_start, 97)
        self.assertEqual(parent_end, 98)

    def test_neg_overlap_right_with_gaps(self): # Case 1.2
        parent_start, parent_end = convert_coord(self.neg_s_info3, self.ref_info3, ref_start=0, ref_end=66)
        self.assertEqual(parent_start, 62)
        self.assertEqual(parent_end, 73)

    def test_neg_spanning_with_gaps(self): # Case 2.2
        parent_start, parent_end = convert_coord(self.neg_s_info3, self.ref_info3, ref_start=0, ref_end=200)
        self.assertEqual(parent_start, 62)
        self.assertEqual(parent_end, 97)

    def test_neg_fully_contained_with_gaps(self): # Case 3.2
        parent_start, parent_end = convert_coord(self.neg_s_info3, self.ref_info3, ref_start=66, ref_end=79)
        self.assertEqual(parent_start, 73)
        self.assertEqual(parent_end, 87)

    def test_neg_overlap_left_with_gaps(self): # Case 4.2
        parent_start, parent_end = convert_coord(self.neg_s_info3, self.ref_info3, ref_start=80, ref_end=200)
        self.assertEqual(parent_start, 88)
        self.assertEqual(parent_end, 97)

    def test_neg_fully_contained_start_gap(self): # Case 3.3
        parent_start, parent_end = convert_coord(self.neg_s_info3, self.ref_info3, ref_start=60, ref_end=79)
        self.assertEqual(parent_start, 67)
        self.assertEqual(parent_end, 87)

    def test_neg_overlap_left_start_gap(self): # Case 4.3
        parent_start, parent_end = convert_coord(self.neg_s_info3, self.ref_info3, ref_start=72, ref_end=200)
        self.assertEqual(parent_start, 82)
        self.assertEqual(parent_end, 97)

    def test_neg_overlap_right_end_gap(self): # Case 1.3
        parent_start, parent_end = convert_coord(self.neg_s_info3, self.ref_info3, ref_start=0, ref_end=73)
        self.assertEqual(parent_start, 62)
        self.assertEqual(parent_end, 82)

    def test_neg_fully_contained_end_gap(self): # Case 3.4
        parent_start, parent_end = convert_coord(self.neg_s_info3, self.ref_info3, ref_start=66, ref_end=83)
        self.assertEqual(parent_start, 73)
        self.assertEqual(parent_end, 90)

    def test_neg_fully_contained_empty(self): # Case 3.5
        parent_start, parent_end = convert_coord(self.neg_s_info4, self.ref_info4, ref_start=62, ref_end=70)
        self.assertEqual(parent_start, None)
        self.assertEqual(parent_end, None)

    def test_ends_on_start(self):
        parent_start, parent_end = convert_coord(self.s_info, self.ref_info, ref_start=0, ref_end=57)
        self.assertEqual(parent_start, None)
        self.assertEqual(parent_end, None)

    # def test_unmapped_ref(self):



if __name__ == "__main__":
    unittest.main()