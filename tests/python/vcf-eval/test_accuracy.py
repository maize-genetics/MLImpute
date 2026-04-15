import pytest
from src.python.vcf_eval.accuracy import *


@pytest.fixture
def truth_vcf():
    return "data/truth.vcf"

@pytest.fixture
def imputed_vcf():
    return "data/imputed.vcf"


def test_write_query_tsv():
    pass

def test_sort_tsv():
    pass

def test_parse_gt():
    # Missing/empty genotypes
    assert parse_gt(".") is None
    assert parse_gt("./.") is None
    assert parse_gt(".|.") is None
    assert parse_gt("") is None  # if your function handles empty strings

    # Homozygous genotypes
    assert parse_gt("0/0") == ['0', '0']  # homozygous reference
    assert parse_gt("1/1") == ['1', '1']  # homozygous alternate
    assert parse_gt("2/2") == ['2', '2']  # homozygous second alternate

    # Heterozygous genotypes
    assert parse_gt("0/1") == ['0', '1']  # het ref/alt
    assert parse_gt("1/0") == ['1', '0']  # het alt/ref (if order matters)
    assert parse_gt("1/2") == ['1', '2']  # het between alternates

    # Phased genotypes (if your function handles them)
    assert parse_gt("0|1") == ['0', '1']
    assert parse_gt("1|0") == ['1', '0']
    assert parse_gt("1|2") == ['1', '2']

    # Partially missing genotypes
    assert parse_gt("./1") is None  # or however your function handles this
    assert parse_gt("1/.") is None
    assert parse_gt(".|1") is None
    assert parse_gt("1|.") is None

    # Higher ploidy (if supported)
    assert parse_gt("0/1/2") == ['0', '1', '2']  # triploid

    # Edge cases
    assert parse_gt("10/11") == ['10', '11']  # multi-digit alleles

def test_gt_to_allele_multiset():
    assert gt_to_allele_multiset(ref="A", alt_field="T", gt="0/0", phase_sensitive=False) == ("A", "A")
    assert gt_to_allele_multiset(ref="A", alt_field="T", gt="1/1", phase_sensitive=False) == ("T", "T")
    assert gt_to_allele_multiset(ref="A", alt_field="T", gt="0/1", phase_sensitive=False) == ("A", "T")
    assert gt_to_allele_multiset(ref="A", alt_field="T", gt="1/0", phase_sensitive=False) == ("A", "T")
    assert gt_to_allele_multiset(ref="A", alt_field="T", gt="1/0", phase_sensitive=True) == ("T", "A")
    assert gt_to_allele_multiset(ref="C", alt_field="G", gt="1|0", phase_sensitive=True) == ("G", "C")

    # Higher ploidy
    assert gt_to_allele_multiset("A", "T,C", "0/1/2") == ("A", "C", "T")  # sorted
    assert gt_to_allele_multiset(ref="A", alt_field="T,C", gt="2/1/0", phase_sensitive=False) == ("A", "C", "T")
    assert gt_to_allele_multiset("A", "T,C", "2/1/0", phase_sensitive=True) == ("C", "T", "A")  # preserves order

    # Edge case: no alternates (monomorphic site)
    result = gt_to_allele_multiset("A", ".", "0/0")
    assert result == ("A", "A")

    result = gt_to_allele_multiset("A", "", "0/0")
    assert result == ("A", "A")

    # Missing genotypes should return None
    assert gt_to_allele_multiset("A", "T", ".") is None
    assert gt_to_allele_multiset("A", "T", "./.") is None
    assert gt_to_allele_multiset("A", "T", ".|.") is None
    assert gt_to_allele_multiset("A", "T", "./1") is None
    assert gt_to_allele_multiset("A", "T", "1/.") is None

def test_allele_multiset_score():
    # Perfect matches
    assert allele_multiset_score(("A", "T"), ("A", "T")) == 1.0
    assert allele_multiset_score(("A", "T"), ("A", "T"), phase_sensitive=True) == 1.0
    assert allele_multiset_score(("A", "T"), ("T", "A")) == 1.0  # order shouldn't matter
    assert allele_multiset_score(("G", "G"), ("G", "G")) == 1.0  # homozygous match
    assert allele_multiset_score(("G", "G"), ("G", "G"), phase_sensitive=True) == 1.0  # homozygous match

    # Complete mismatches
    assert allele_multiset_score(("A", "T"), ("C", "G")) == 0.0
    assert allele_multiset_score(("A", "A"), ("T", "T")) == 0.0  # homozygous mismatch
    assert allele_multiset_score(("A", "T"), ("T", "A"), phase_sensitive=True) == 0.0  # order matters
    assert allele_multiset_score(("A", "T"), ("T", "C"), phase_sensitive=True) == 0.0  # order matters

    # Partial matches (diploid)
    assert allele_multiset_score(("A", "T"), ("A", "C")) == 0.5  # one allele matches
    assert allele_multiset_score(("A", "T"), ("A", "C"), phase_sensitive=True) == 0.5  # one allele matches
    assert allele_multiset_score(("A", "T"), ("C", "T")) == 0.5  # one allele matches
    assert allele_multiset_score(("A", "T"), ("C", "T"), phase_sensitive=True) == 0.5  # one allele matches
    assert allele_multiset_score(("A", "A"), ("A", "T")) == 0.5  # one of two A's matches
    assert allele_multiset_score(("A", "A"), ("A", "T"), phase_sensitive=True) == 0.5  # one of two A's matches

    # Different ploidy scenarios

    # Haploid truth vs diploid imputed
    assert allele_multiset_score(("A",), ("A", "A")) == 1.0  # 1 match out of 1 truth allele
    assert allele_multiset_score(("A",), ("A", "A"), phase_sensitive=True) == 1.0
    assert allele_multiset_score(("A",), ("A", "T")) == 1.0  # 1 match out of 1 truth allele
    assert allele_multiset_score(("A",), ("A", "T"), phase_sensitive=True) == 1.0
    assert allele_multiset_score(("A",), ("T", "A"), phase_sensitive=True) == 0.0
    assert allele_multiset_score(("A",), ("T", "C")) == 0.0  # 0 matches out of 1 truth allele
    assert allele_multiset_score(("A",), ("T", "C"), phase_sensitive=True) == 0.0

    # Diploid truth vs haploid imputed
    assert allele_multiset_score(("A", "T"), ("A",)) == 0.5  # 1 match out of 2 truth alleles
    assert allele_multiset_score(("A", "T"), ("A",), phase_sensitive=True) == 0.5
    assert allele_multiset_score(("A", "T"), ("T",), phase_sensitive=True) == 0.0 # I think should be 0.5
    assert allele_multiset_score(("A", "T"), ("C",)) == 0.0  # 0 matches out of 2 truth alleles
    assert allele_multiset_score(("A", "A"), ("A",)) == 0.5  # 1 match out of 2 truth alleles

    # Triploid scenarios
    assert allele_multiset_score(("A", "T", "C"), ("A", "T", "C")) == 1.0  # perfect match
    assert allele_multiset_score(("A", "T", "C"), ("A", "T", "G")) == 2.0 / 3.0  # 2 out of 3 match
    assert allele_multiset_score(("A", "T", "C"), ("A", "G", "T"), phase_sensitive=True) == 1.0 / 3.0
    assert allele_multiset_score(("A", "T", "C"), ("A", "G", "G")) == 1.0 / 3.0  # 1 out of 3 match
    assert allele_multiset_score(("A", "T", "C"), ("G", "G", "G")) == 0.0  # 0 out of 3 match

    # Triploid truth vs diploid imputed
    assert allele_multiset_score(("A", "A", "T"), ("A", "A")) == 2.0 / 3.0  # 2 A matches out of 3 truth
    assert allele_multiset_score(("A", "T", "C"), ("A", "T")) == 2.0 / 3.0  # 2 matches out of 3 truth

    # Complex multiset scenarios (testing Counter intersection logic)
    assert allele_multiset_score(("A", "A", "T"), ("A", "T", "T")) == 2.0 / 3.0  # 1 A + 1 T match
    assert allele_multiset_score(("A", "A", "A"), ("A", "A", "T")) == 2.0 / 3.0  # 2 A's match
    assert allele_multiset_score(("A", "T", "T"), ("T", "C", "C")) == 1.0 / 3.0  # 1 T matches

    # Edge case: empty truth (should return 0.0)
    assert allele_multiset_score((), ("A", "T")) == 0.0
    assert allele_multiset_score((), ()) == 0.0

    # Edge case: empty imputed but non-empty truth
    assert allele_multiset_score(("A", "T"), ()) == 0.0

    # Single allele cases
    assert allele_multiset_score(("A",), ("A",)) == 1.0
    assert allele_multiset_score(("A",), ("T",)) == 0.0

    # Complex allele strings (indels)
    assert allele_multiset_score(("ATG", "A"), ("ATG", "A")) == 1.0
    assert allele_multiset_score(("ATG", "A"), ("ATG", "C")) == 0.5
    assert allele_multiset_score(("AAG", "C"), ("ATG", "C")) == 0.5
    assert allele_multiset_score(("ATGCCC", "DEL"), ("ATGCCC", "DEL")) == 1.0

    # Many copies of same allele
    truth_many_a = ("A", "A", "A", "A")  # 4 copies of A
    imputed_some_a = ("A", "A", "T", "T")  # 2 copies of A
    assert allele_multiset_score(truth_many_a, imputed_some_a) == 0.5  # 2 out of 4 A's match

    # More complex multiset intersections
    truth = ("A", "A", "B", "B", "C")  # 2 A's, 2 B's, 1 C
    imputed = ("A", "B", "B", "C", "C")  # 1 A, 2 B's, 2 C's
    # Intersection: 1 A (min(2,1)), 2 B's (min(2,2)), 1 C (min(1,2)) = 4 total
    expected = 4.0 / 5.0  # 4 matches out of 5 truth alleles
    assert allele_multiset_score(truth, imputed) == expected

def test_iter_records():
    pass

def test_compare_sorted():
    pass

def test_initialize_counts():
    assert initialize_counts() == {
        "truth_records": 0,
        "imputed_records": 0,
        "site_key_matches": 0,
        "missing_in_imputed_sites": 0,
        "extra_in_imputed_sites": 0,
        "both_missing_gt": 0,
        "truth_missing_gt": 0,
        "imputed_missing_gt": 0,
        "gt_unparseable": 0,
        "compared_sites": 0,
        "gt_allele_matches": 0,
        "gt_allele_mismatches": 0,
        "partial_credit_sum": 0.0,
        "examples": [],
        "af_bin_total": None,
        "af_bin_correct": None,
        "af_bins": None,
        "het_bin_total": None,
        "het_bin_correct": None,
    }

def test_write_sites():
    pass

def test_handle_missing_in_imputed():
    pass

def test_handle_extra_in_imputed():
    pass

def test_handle_matched_sites():
    pass

def test_extract_allele_frequency():
    assert extract_allele_frequency(t_info="AN=10;AC=2") == (2, 10)
    assert extract_allele_frequency(t_info="AC=2;AN=10") == (2, 10)
    assert extract_allele_frequency(t_info="AN=10") == (None, 10)
    assert extract_allele_frequency(t_info="AC=2") == (2, None)
    assert extract_allele_frequency(t_info="AN=10AC=2") == (None, None)
    assert extract_allele_frequency(t_info=".") == (None, None)

def test_update_frequency_bins():
    pass

def test_handle_missing_genotypes():
    pass

def test_next_or_none():
    pass