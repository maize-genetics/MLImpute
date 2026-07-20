"""PyTorch port of phg_v2's PS4G Viterbi path finder (PathFinderHMMPS4G.kt et al.).

Builds haploid/diploid emission and transition tensors directly from a PS4G file
(see docs/ps4g_specifications.md) and decodes the most likely path with
python.hmm.viterbi.viterbi_decode, which runs on CPU or GPU depending on the
tensors' device.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from python.hmm.viterbi import viterbi_decode
from python.ps4g_io.ps4g import load_ps4g_file

# Mirrors the Double.MIN_VALUE floor applied to linear probabilities by
# HaploidPS4GEmissionProbability.kt before taking the log.
LN_MIN_PROBABILITY = math.log(5e-324)


@dataclass(frozen=True)
class GameteSet:
    indices: Tuple[int, ...]
    count: int


def resolve_device(device: Optional[str] = "auto") -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def parse_gamete_index_map(ps4g_file: str) -> Dict[int, str]:
    """Parse '#<gamete>\\t<index>\\t<count>' header lines into {index: gamete_name}.

    Keeps the full gamete identifier (e.g. "lineA:0"), unlike
    ps4g_io.ps4g.build_index_lookup which strips the ":<gameteIndex>" suffix -
    the Kotlin reader (Ps4gFileReader.kt) keeps the full identifier and the
    Kotlin CLI output depends on it.
    """
    gamete_index_map: Dict[int, str] = {}
    with open(ps4g_file) as fh:
        for line in fh:
            if not line.startswith("#"):
                continue
            stripped = line.strip()
            if ":" not in stripped or "\t" not in stripped:
                continue
            parts = stripped[1:].split("\t")
            if len(parts) != 3:
                continue
            name, idx_str, _count_str = parts
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            gamete_index_map[idx] = name
    if not gamete_index_map:
        raise ValueError(
            f"{ps4g_file} did not contain any '#<gamete>\\t<index>\\t<count>' header lines."
        )
    return gamete_index_map


def build_contig_readmap(ps4g_file: str) -> Dict[str, Dict[int, List[GameteSet]]]:
    """Build {contig: {binnedPos: [GameteSet, ...]}}, mirroring Ps4gFileReader.kt."""
    with open(ps4g_file) as fh:
        first_line = fh.readline().strip()
        second_line = fh.readline().strip()
    if first_line != "#PS4G":
        raise ValueError(f"{ps4g_file} is not a valid PS4G file. First line is '{first_line}'")
    if second_line.lower() != "#version=2.0":
        raise ValueError(f"Second row of {ps4g_file} is not a valid PS4G version string: '{second_line}'")

    df = load_ps4g_file(ps4g_file)
    contig_readmap: Dict[str, Dict[int, List[GameteSet]]] = {}
    for row in df.itertuples(index=False):
        contig_map = contig_readmap.setdefault(row.refContig, {})
        contig_map.setdefault(int(row.refPosBinned), []).append(
            GameteSet(indices=tuple(int(i) for i in row.gameteSet), count=int(row.count))
        )
    return contig_readmap


def binom_logpmf(k: torch.Tensor, n: torch.Tensor, p: float) -> torch.Tensor:
    """Vectorized natural-log binomial pmf: ln( C(n,k) p^k (1-p)^(n-k) )."""
    p_t = torch.as_tensor(p, dtype=k.dtype, device=k.device)
    log_p = torch.log(p_t)
    log_1mp = torch.log1p(-p_t)
    log_binom_coeff = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    return log_binom_coeff + k * log_p + (n - k) * log_1mp


def build_counts_and_cooccurrence(
    readmap_for_contig: Dict[int, List[GameteSet]],
    sorted_positions: Sequence[int],
    index_list: Sequence[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-position read-support tensors restricted to `index_list`, on the fly
    from the PS4G read map (this is the "build the numpy array on the fly" step).

    Returns:
      total_counts: [T]     -- sum of all entry counts at each position (binomial n)
      counts:       [T, K]  -- counts[t,k] = counts of entries containing index_list[k]
      co_occurrence:[T,K,K] -- counts of entries containing both index_list[i] and
                               index_list[j] (diagonal == counts); used to derive
                               "i or j" counts via inclusion-exclusion for diploid
                               heterozygous emissions.
    """
    T = len(sorted_positions)
    K = len(index_list)
    index_position = {gamete_index: k for k, gamete_index in enumerate(index_list)}

    total_counts = np.zeros(T, dtype=np.float64)
    counts = np.zeros((T, K), dtype=np.float64)
    co_occurrence = np.zeros((T, K, K), dtype=np.float64)

    for t, pos in enumerate(sorted_positions):
        entries = readmap_for_contig.get(pos) or []
        if not entries:
            continue
        e = len(entries)
        membership = np.zeros((e, K), dtype=np.float64)
        entry_counts = np.zeros(e, dtype=np.float64)
        for row, gamete_set in enumerate(entries):
            entry_counts[row] = gamete_set.count
            for gamete_index in gamete_set.indices:
                k = index_position.get(gamete_index)
                if k is not None:
                    membership[row, k] = 1.0
        total_counts[t] = entry_counts.sum()
        weighted = entry_counts[:, None] * membership
        counts[t] = weighted.sum(axis=0)
        co_occurrence[t] = membership.T @ weighted

    return (
        torch.tensor(total_counts, dtype=torch.float64, device=device),
        torch.tensor(counts, dtype=torch.float64, device=device),
        torch.tensor(co_occurrence, dtype=torch.float64, device=device),
    )


def build_haploid_emissions(
    readmap_for_contig: Dict[int, List[GameteSet]],
    sorted_positions: Sequence[int],
    gamete_index_list: Sequence[int],
    p_correct: float,
    device: torch.device,
) -> torch.Tensor:
    """T x N ln P(obs|gamete), matching HaploidPS4GEmissionProbability.kt."""
    total_counts, counts, _ = build_counts_and_cooccurrence(
        readmap_for_contig, sorted_positions, gamete_index_list, device
    )
    n = total_counts.unsqueeze(1).expand_as(counts)
    log_em = binom_logpmf(counts, n, p_correct)
    return torch.clamp_min(log_em, LN_MIN_PROBABILITY)


def build_haploid_transitions(n_gametes: int, p_same: float, device: torch.device) -> torch.Tensor:
    p_switch = (1.0 - p_same) / (n_gametes - 1) if n_gametes > 1 else 0.0
    log_same = math.log(p_same)
    log_switch = math.log(p_switch) if p_switch > 0 else float("-inf")
    log_A = torch.full((n_gametes, n_gametes), log_switch, dtype=torch.float64, device=device)
    log_A.fill_diagonal_(log_same)
    return log_A


def find_haploid_path(
    contig: str,
    gamete_index_map: Dict[int, str],
    readmap_for_contig: Dict[int, List[GameteSet]],
    p_correct: float,
    p_same: float,
    device: Optional[torch.device] = None,
) -> List[Tuple[int, str]]:
    device = device or resolve_device()
    sorted_positions = sorted(readmap_for_contig.keys())
    gamete_indices = sorted(gamete_index_map.keys())
    n = len(gamete_indices)
    if n == 0:
        raise ValueError("gamete_index_map must not be empty")

    log_em = build_haploid_emissions(readmap_for_contig, sorted_positions, gamete_indices, p_correct, device)
    log_A = build_haploid_transitions(n, p_same, device)
    log_start = torch.zeros(n, dtype=torch.float64, device=device)

    path_indices = viterbi_decode(log_em, log_A, log_start)
    return [
        (pos, gamete_index_map[gamete_indices[state_idx]])
        for pos, state_idx in zip(sorted_positions, path_indices)
    ]


def diploid_ordered_states(parent_list: Sequence[int]) -> List[Tuple[int, int]]:
    """All ordered (a,b) pairs over the (sorted) parent list, mirroring the nested
    `for index1 in parentSet { for index2 in parentSet { ... } }` loop in
    PathFinderHMMPS4G.diploidViterbi, which tracks the two homologous chromosome
    copies as an ordered pair rather than an unordered one.
    """
    parents = sorted(parent_list)
    return [(a, b) for a in parents for b in parents]


def build_diploid_emissions(
    readmap_for_contig: Dict[int, List[GameteSet]],
    sorted_positions: Sequence[int],
    parent_list: Sequence[int],
    p_correct: float,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """T x S ln P(obs|parent pair), matching DiploidEmissionProbabilityForLikelyParents.kt.

    S = len(parent_list)**2 ordered pairs; the emission itself is symmetric in
    (a,b) vs (b,a) (as in the Kotlin `parentIndicesToArrayIndex` normalization).
    """
    parents = sorted(parent_list)
    total_counts, counts, co_occurrence = build_counts_and_cooccurrence(
        readmap_for_contig, sorted_positions, parents, device
    )
    # Inclusion-exclusion: count(i or j) = count(i) + count(j) - count(i and j).
    # For i == j this reduces to count(i), matching the homozygous case.
    counts_i = counts.unsqueeze(2)
    counts_j = counts.unsqueeze(1)
    or_counts = counts_i + counts_j - co_occurrence
    n = total_counts.view(-1, 1, 1).expand_as(or_counts)
    log_pair = binom_logpmf(or_counts, n, p_correct)

    states = diploid_ordered_states(parents)
    parent_pos = {g: i for i, g in enumerate(parents)}
    col_i = torch.tensor([parent_pos[a] for a, _b in states], device=device)
    col_j = torch.tensor([parent_pos[b] for _a, b in states], device=device)
    log_em = log_pair[:, col_i, col_j]
    return log_em, states


def _diploid_transition_prob_f0(frm: Tuple[int, int], to: Tuple[int, int], p_same: float, p_switch: float) -> float:
    a, b = frm
    c, d = to
    if a == c:
        return p_same * p_same if b == d else p_switch * p_same
    return p_switch * p_same if b == d else p_switch * p_switch


def _diploid_transition_prob_f1(frm: Tuple[int, int], to: Tuple[int, int], p_same: float, p_switch: float) -> float:
    a, b = frm
    c, d = to
    if a == b:
        if c == d:
            return p_same if a == c else p_switch
        return 0.0
    if c == d:
        return p_switch if (a == c or b == d) else p_switch * p_switch
    return 0.0


def build_diploid_transitions(
    parent_list: Sequence[int],
    p_same: float,
    inbreed_coef: float,
    n_genomes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """S x S ln P(to|from), matching DiploidTransitionWithInbreeding.kt. p_switch is
    computed from n_genomes (all gametes), not the restricted parent count -
    matching the Kotlin constructor which is given gameteIndexSet.size.
    """
    states = diploid_ordered_states(parent_list)
    p_switch = (1.0 - p_same) / (n_genomes - 1) if n_genomes > 1 else 0.0

    S = len(states)
    log_trans = torch.empty((S, S), dtype=torch.float64, device=device)
    for i, frm in enumerate(states):
        for j, to in enumerate(states):
            if inbreed_coef == 0.0:
                p = _diploid_transition_prob_f0(frm, to, p_same, p_switch)
            elif inbreed_coef == 1.0:
                p = _diploid_transition_prob_f1(frm, to, p_same, p_switch)
            else:
                p = (1.0 - inbreed_coef) * _diploid_transition_prob_f0(frm, to, p_same, p_switch) + \
                    inbreed_coef * _diploid_transition_prob_f1(frm, to, p_same, p_switch)
            log_trans[i, j] = math.log(p) if p > 0 else float("-inf")
    return log_trans, states


def diploid_log_start(states: Sequence[Tuple[int, int]], n_genomes: int, device: torch.device) -> torch.Tensor:
    homo = math.log(1.0 / n_genomes)
    hetero = math.log(1.0 / (n_genomes * n_genomes - n_genomes))
    return torch.tensor(
        [homo if a == b else hetero for a, b in states], dtype=torch.float64, device=device
    )


def find_diploid_path(
    contig: str,
    gamete_index_map: Dict[int, str],
    readmap_for_contig: Dict[int, List[GameteSet]],
    parent_set: Sequence[int],
    p_correct: float,
    p_same: float,
    inbreed_coef: float,
    device: Optional[torch.device] = None,
) -> List[Tuple[int, str, str]]:
    device = device or resolve_device()
    sorted_positions = sorted(readmap_for_contig.keys())
    n_genomes = len(gamete_index_map)
    parents = sorted(set(parent_set))

    log_em, states = build_diploid_emissions(readmap_for_contig, sorted_positions, parents, p_correct, device)
    log_trans, states2 = build_diploid_transitions(parents, p_same, inbreed_coef, n_genomes, device)
    assert states == states2
    log_start = diploid_log_start(states, n_genomes, device)

    path_indices = viterbi_decode(log_em, log_trans, log_start)
    result = []
    for pos, state_idx in zip(sorted_positions, path_indices):
        a, b = states[state_idx]
        result.append((pos, gamete_index_map[a], gamete_index_map[b]))
    return result


def most_likely_parents(
    contig_readmap: Dict[str, Dict[int, List[GameteSet]]],
    contigs: Sequence[str],
    n_parents: int,
) -> set:
    """Port of MostLikelyPs4gParents.bestParents: greedily pick the highest-count
    gamete, then the highest-count gamete once the first choice is excluded, and
    repeat, alternating with the complement of the very first choice.
    """
    if n_parents <= 1:
        raise ValueError("Number of best parents must be greater than 1.")

    gamete_sets: List[GameteSet] = []
    for contig in contigs:
        readmap_for_contig = contig_readmap.get(contig)
        if readmap_for_contig is None:
            raise ValueError(
                f"contigReadMap is null for contig = {contig} caused by an incorrect "
                "contig name passed to most_likely_parents."
            )
        for entries in readmap_for_contig.values():
            gamete_sets.extend(entries)

    def gamete_counts(sets: List[GameteSet]) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for gs in sets:
            for idx in gs.indices:
                counts[idx] = counts.get(idx, 0) + gs.count
        return counts

    def best_from_filtered(sets: List[GameteSet], excluded: int) -> Optional[int]:
        filtered = [gs for gs in sets if excluded not in gs.indices]
        counts = gamete_counts(filtered)
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

    counts = gamete_counts(gamete_sets)
    if not counts:
        raise ValueError("chosen Parent is null in most_likely_parents.")
    chosen_parent = max(counts.items(), key=lambda kv: kv[1])[0]
    best_parents: List[int] = [chosen_parent]
    counts.pop(chosen_parent, None)

    next_parent = best_from_filtered(gamete_sets, chosen_parent)
    if next_parent is None:
        raise ValueError("next Parent is null in most_likely_parents.")
    best_parents.append(next_parent)
    counts.pop(next_parent, None)

    while len(best_parents) < n_parents:
        if not counts:
            raise ValueError(f"Could only choose {len(best_parents)} best parents.")
        highest = max(counts.items(), key=lambda kv: kv[1])[0]
        best_parents.append(highest)
        counts.pop(highest, None)

        if len(best_parents) < n_parents:
            compliment = best_from_filtered(gamete_sets, chosen_parent)
            if compliment is None:
                raise ValueError(f"Could only choose {len(best_parents)} best parents.")
            best_parents.append(compliment)
            counts.pop(compliment, None)

    return set(best_parents)
