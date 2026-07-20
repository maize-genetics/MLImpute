import math

import pytest
import torch

from python.hmm.ps4g_hmm import (
    GameteSet,
    LN_MIN_PROBABILITY,
    binom_logpmf,
    build_contig_readmap,
    build_counts_and_cooccurrence,
    build_diploid_emissions,
    build_diploid_transitions,
    build_haploid_emissions,
    build_haploid_transitions,
    diploid_log_start,
    diploid_ordered_states,
    find_diploid_path,
    find_haploid_path,
    most_likely_parents,
    parse_gamete_index_map,
    resolve_device,
    _diploid_transition_prob_f0,
    _diploid_transition_prob_f1,
)


def write_ps4g_file(path, gamete_lines, data_lines, total_counts=None):
    """gamete_lines: list of (name, index, count); data_lines: list of (gameteSet_csv, contig, pos, count)."""
    if total_counts is None:
        total_counts = sum(int(d[3]) for d in data_lines)
    with open(path, "w") as fh:
        fh.write("#PS4G\n")
        fh.write("#version=2.0\n")
        fh.write("#Command: test\n")
        fh.write(f"#TotalUniqueCounts: {total_counts}\n")
        fh.write("#gamete\tgameteIndex\tcount\n")
        for name, idx, count in gamete_lines:
            fh.write(f"#{name}\t{idx}\t{count}\n")
        fh.write("gameteSet\trefContig\trefPosBinned\tcount\n")
        for gamete_set, contig, pos, count in data_lines:
            fh.write(f"{gamete_set}\t{contig}\t{pos}\t{count}\n")
    return str(path)


class TestBinomLogpmf:
    def test_matches_hand_computed_values(self):
        cases = [(10, 3, 0.5), (15, 8, 0.99), (1, 1, 0.6), (1, 0, 0.6), (4, 0, 0.98)]
        for n, k, p in cases:
            expected = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) \
                + k * math.log(p) + (n - k) * math.log(1 - p)
            got = binom_logpmf(
                torch.tensor(float(k), dtype=torch.float64), torch.tensor(float(n), dtype=torch.float64), p
            ).item()
            assert got == pytest.approx(expected, abs=1e-9)

    def test_zero_trials_zero_successes_is_ln_one(self):
        # Mirrors the "no reads -> emission = 0.0" branch of HaploidPS4GEmissionProbability.kt,
        # which falls out naturally from the binomial formula without special-casing.
        got = binom_logpmf(torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64), 0.98).item()
        assert got == pytest.approx(0.0, abs=1e-12)


class TestGameteIndexMapParsing:
    def test_parse_gamete_index_map_keeps_full_identifier(self, tmp_path):
        ps4g_file = write_ps4g_file(
            tmp_path / "sample.ps4g",
            gamete_lines=[("lineA:0", 0, 10), ("lineB:0", 1, 5)],
            data_lines=[("0", "chr1", 1, 10), ("1", "chr1", 2, 5)],
        )
        gamete_index_map = parse_gamete_index_map(ps4g_file)
        assert gamete_index_map == {0: "lineA:0", 1: "lineB:0"}

    def test_rejects_missing_header(self, tmp_path):
        bad_file = tmp_path / "bad.ps4g"
        bad_file.write_text("not a ps4g file\n")
        with pytest.raises(ValueError):
            build_contig_readmap(str(bad_file))


class TestBuildContigReadmap:
    def test_matches_ps4g_file_reader_structure(self, tmp_path):
        ps4g_file = write_ps4g_file(
            tmp_path / "sample.ps4g",
            gamete_lines=[("sampleA:0", 0, 5), ("sampleA:1", 1, 3), ("sampleB:0", 2, 5), ("sampleB:1", 3, 3)],
            data_lines=[("0,2", "1", 100, 5), ("1,3", "1", 200, 3)],
        )
        contig_readmap = build_contig_readmap(ps4g_file)
        assert set(contig_readmap.keys()) == {"1"}
        reads = contig_readmap["1"]
        assert len(reads) == 2
        assert reads[100] == [GameteSet(indices=(0, 2), count=5)]
        assert reads[200] == [GameteSet(indices=(1, 3), count=3)]


class TestHaploidEmissions:
    def test_hand_example(self):
        # Position 1: gamete 0 gets 4 reads out of 6 total (2 shared with gamete 1).
        readmap = {
            1: [GameteSet(indices=(0,), count=4), GameteSet(indices=(0, 1), count=2)],
        }
        device = torch.device("cpu")
        log_em = build_haploid_emissions(readmap, [1], [0, 1], p_correct=0.9, device=device)
        assert log_em.shape == (1, 2)

        n = 6
        k0 = 6  # gamete 0 appears in both entries: 4 + 2
        k1 = 2  # gamete 1 appears only in the second entry
        expected0 = max(
            math.lgamma(n + 1) - math.lgamma(k0 + 1) - math.lgamma(n - k0 + 1) + k0 * math.log(0.9) + (n - k0) * math.log(0.1),
            LN_MIN_PROBABILITY,
        )
        expected1 = max(
            math.lgamma(n + 1) - math.lgamma(k1 + 1) - math.lgamma(n - k1 + 1) + k1 * math.log(0.9) + (n - k1) * math.log(0.1),
            LN_MIN_PROBABILITY,
        )
        assert log_em[0, 0].item() == pytest.approx(expected0, abs=1e-6)
        assert log_em[0, 1].item() == pytest.approx(expected1, abs=1e-6)

    def test_empty_position_is_zero(self):
        readmap = {5: []}
        log_em = build_haploid_emissions(readmap, [5], [0, 1, 2], p_correct=0.9, device=torch.device("cpu"))
        assert torch.allclose(log_em, torch.zeros_like(log_em))


class TestDiploidEmissions:
    """Anchored on the exact counts from DiploidPS4GEmissionProbabilityTest.kt (testIndexCountsForOneIndex /
    testIndexCountsForTwoIndexes), which share the same membership-counting semantics as
    DiploidEmissionProbabilityForLikelyParents (the class actually used by the path finder).
    """

    @pytest.fixture
    def readmap(self):
        return {
            1: [
                GameteSet(indices=(0, 1), count=4),
                GameteSet(indices=(2, 3), count=2),
                GameteSet(indices=(0, 1, 2), count=3),
                GameteSet(indices=(0, 1, 3), count=1),
                GameteSet(indices=(1, 3), count=5),
            ]
        }

    def test_counts_and_cooccurrence_diagonal_matches_one_index_counts(self, readmap):
        total_counts, counts, co_occurrence = build_counts_and_cooccurrence(
            readmap, [1], [0, 1, 2, 3], torch.device("cpu")
        )
        assert total_counts[0].item() == 15
        assert counts[0].tolist() == [8, 13, 5, 8]
        assert torch.diagonal(co_occurrence[0]).tolist() == [8, 13, 5, 8]

    def test_or_counts_match_two_index_counts(self, readmap):
        _total, counts, co_occurrence = build_counts_and_cooccurrence(
            readmap, [1], [0, 1, 2, 3], torch.device("cpu")
        )
        counts_i = counts.unsqueeze(2)
        counts_j = counts.unsqueeze(1)
        or_counts = (counts_i + counts_j - co_occurrence)[0]
        assert or_counts[0, 1].item() == 13
        assert or_counts[0, 2].item() == 10
        assert or_counts[0, 3].item() == 15

    def test_emissions_use_binomial_of_or_counts(self, readmap):
        log_em, states = build_diploid_emissions(readmap, [1], [0, 1, 2, 3], p_correct=0.95, device=torch.device("cpu"))
        n = 15
        expected_homozygous_0 = math.lgamma(n + 1) - math.lgamma(9) - math.lgamma(n - 8 + 1) + 8 * math.log(0.95) + 7 * math.log(0.05)
        idx = states.index((0, 0))
        assert log_em[0, idx].item() == pytest.approx(expected_homozygous_0, abs=1e-6)

        expected_hetero_01 = math.lgamma(n + 1) - math.lgamma(14) - math.lgamma(n - 13 + 1) + 13 * math.log(0.95) + 2 * math.log(0.05)
        idx01 = states.index((0, 1))
        idx10 = states.index((1, 0))
        assert log_em[0, idx01].item() == pytest.approx(expected_hetero_01, abs=1e-6)
        # Emission is symmetric under swapping the two chromosome copies.
        assert log_em[0, idx01].item() == pytest.approx(log_em[0, idx10].item(), abs=1e-9)


class TestTransitions:
    def test_haploid_transition_diagonal_and_off_diagonal(self):
        log_A = build_haploid_transitions(3, p_same=0.9, device=torch.device("cpu"))
        assert log_A.shape == (3, 3)
        assert log_A[0, 0].item() == pytest.approx(math.log(0.9))
        assert log_A[0, 1].item() == pytest.approx(math.log(0.05))
        assert log_A[1, 2].item() == pytest.approx(math.log(0.05))

    def test_diploid_ordered_states_shape(self):
        states = diploid_ordered_states([2, 0, 1])
        assert states == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    def test_diploid_transition_f0(self):
        # p_same=0.9, n_genomes=3 -> p_switch=0.05
        p_same, p_switch = 0.9, 0.05
        assert _diploid_transition_prob_f0((0, 0), (0, 0), p_same, p_switch) == pytest.approx(0.81)
        assert _diploid_transition_prob_f0((0, 0), (0, 1), p_same, p_switch) == pytest.approx(0.045)
        assert _diploid_transition_prob_f0((0, 1), (1, 0), p_same, p_switch) == pytest.approx(0.0025)

    def test_diploid_transition_f1(self):
        p_same, p_switch = 0.9, 0.05
        # from homozygous
        assert _diploid_transition_prob_f1((0, 0), (0, 0), p_same, p_switch) == pytest.approx(0.9)
        assert _diploid_transition_prob_f1((0, 0), (1, 1), p_same, p_switch) == pytest.approx(0.05)
        assert _diploid_transition_prob_f1((0, 0), (0, 1), p_same, p_switch) == 0.0
        # from heterozygous
        assert _diploid_transition_prob_f1((0, 1), (0, 0), p_same, p_switch) == pytest.approx(0.05)
        assert _diploid_transition_prob_f1((0, 1), (1, 1), p_same, p_switch) == pytest.approx(0.05)
        assert _diploid_transition_prob_f1((0, 1), (2, 2), p_same, p_switch) == pytest.approx(0.0025)
        assert _diploid_transition_prob_f1((0, 1), (0, 1), p_same, p_switch) == 0.0

    def test_build_diploid_transitions_uses_full_genome_count_for_p_switch(self):
        # nGenomes (5) differs from the restricted parent list (2) - p_switch must be
        # derived from nGenomes, matching DiploidTransitionWithInbreeding's constructor arg.
        log_trans, states = build_diploid_transitions([0, 1], p_same=0.9, inbreed_coef=0.0, n_genomes=5, device=torch.device("cpu"))
        p_switch = (1 - 0.9) / (5 - 1)
        idx_00 = states.index((0, 0))
        idx_01 = states.index((0, 1))
        assert log_trans[idx_00, idx_00].item() == pytest.approx(math.log(0.9 * 0.9))
        assert log_trans[idx_00, idx_01].item() == pytest.approx(math.log(p_switch * 0.9))

    def test_diploid_log_start(self):
        states = [(0, 0), (0, 1), (1, 1)]
        log_start = diploid_log_start(states, n_genomes=3, device=torch.device("cpu"))
        assert log_start[0].item() == pytest.approx(math.log(1 / 3))
        assert log_start[1].item() == pytest.approx(math.log(1 / 6))
        assert log_start[2].item() == pytest.approx(math.log(1 / 3))


class TestMostLikelyParents:
    def test_deterministic_top_two(self):
        # Gametes 0 and 1 each have strong *independent* support (never sharing a
        # gameteSet with each other), so gamete 1's evidence survives the second
        # step's filtering-out of every gameteSet that contains the first pick.
        # Gametes 2,3,4 are rare and never co-occur with 0 or 1.
        readmap = {pos: [GameteSet(indices=(0,), count=10)] for pos in range(5)}
        readmap.update({pos: [GameteSet(indices=(1,), count=10)] for pos in range(5, 10)})
        readmap[10] = [GameteSet(indices=(2,), count=3)]
        readmap[11] = [GameteSet(indices=(3,), count=3)]
        readmap[12] = [GameteSet(indices=(4,), count=3)]

        contig_readmap = {"chr1": readmap}
        best = most_likely_parents(contig_readmap, ["chr1"], 2)
        assert best == {0, 1}

    def test_requires_more_than_one_parent(self):
        with pytest.raises(ValueError):
            most_likely_parents({"chr1": {0: [GameteSet(indices=(0,), count=1)]}}, ["chr1"], 1)

    def test_missing_contig_raises(self):
        with pytest.raises(ValueError):
            most_likely_parents({"chr1": {}}, ["chr2"], 2)


class TestHaploidPathFinding:
    def test_all_reads_support_gamete_zero(self):
        readmap = {pos: [GameteSet(indices=(0,), count=10)] for pos in range(1, 6)}
        gamete_index_map = {0: "lineA:0", 1: "lineB:0"}
        path = find_haploid_path("chr1", gamete_index_map, readmap, p_correct=0.99, p_same=0.9999, device=torch.device("cpu"))
        assert [name for _pos, name in path] == ["lineA:0"] * 5

    def test_recombination_switches_path(self):
        readmap = {}
        for pos in range(1, 5):
            readmap[pos] = [GameteSet(indices=(0,), count=20)]
        for pos in range(5, 9):
            readmap[pos] = [GameteSet(indices=(1,), count=20)]
        gamete_index_map = {0: "lineA:0", 1: "lineB:0"}
        path = find_haploid_path("chr1", gamete_index_map, readmap, p_correct=0.99, p_same=0.5, device=torch.device("cpu"))
        names = [name for _pos, name in path]
        assert names[:4] == ["lineA:0"] * 4
        assert names[4:] == ["lineB:0"] * 4


class TestDiploidPathFinding:
    def test_inbred_homozygous_path(self):
        readmap = {pos: [GameteSet(indices=(0,), count=10)] for pos in range(1, 6)}
        gamete_index_map = {0: "lineA:0", 1: "lineB:0"}
        path = find_diploid_path(
            "chr1", gamete_index_map, readmap, parent_set={0, 1},
            p_correct=0.99, p_same=0.9999, inbreed_coef=1.0, device=torch.device("cpu"),
        )
        for _pos, name1, name2 in path:
            assert name1 == "lineA:0"
            assert name2 == "lineA:0"


class TestDevice:
    def test_resolve_device_auto_cpu_fallback(self):
        device = resolve_device("cpu")
        assert device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cpu_and_cuda_paths_match(self):
        readmap = {}
        for pos in range(1, 5):
            readmap[pos] = [GameteSet(indices=(0,), count=20)]
        for pos in range(5, 9):
            readmap[pos] = [GameteSet(indices=(1,), count=20)]
        gamete_index_map = {0: "lineA:0", 1: "lineB:0"}

        cpu_path = find_haploid_path("chr1", gamete_index_map, readmap, 0.99, 0.5, device=torch.device("cpu"))
        cuda_path = find_haploid_path("chr1", gamete_index_map, readmap, 0.99, 0.5, device=torch.device("cuda"))
        assert cpu_path == cuda_path
