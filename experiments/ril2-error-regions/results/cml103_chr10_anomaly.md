# CML103's low fleet accuracy: a real chr10 anomaly, amplified by a non-random test split

Follow-up to `oh43_error_clustering_31mer.md`'s Check B, which applied the self-absent
framework fleet-wide and found CML52/Tzi8's poor `nam_baseline.md` accuracy is explained by
elevated *genome-wide* self-absent rate (11.3%/19.9% vs. the fleet's 1.9–5.6% norm) — but
**CML103 (viterbi=0.79) showed a perfectly normal genome-wide self-absent rate (4.09%)**,
leaving its poor accuracy unresolved. This applies the same framework specifically to
CML103 and finds the real cause: not diffuse difficulty, but a sharp, isolated anomaly on
one chromosome, whose effect on the reported score turns out to be amplified by a
non-random detail of the evaluation pipeline itself.

**Scope**: diagnose-only; used the original, preserved per-founder scratch directories from
the `nam_baseline.py` benchmark run (`.../scratchpad/nam_baseline/<Founder>/`, still on disk
for all 24 founders) directly, rather than regenerating — this guarantees byte-for-byte
identical results to the numbers in `results/nam_baseline_results.tsv`.

## Step 1: reproduce the reported numbers exactly

Reproduced CML103's `abs_pct=14.24%` from first principles using only `numpy` (no model,
no torch needed — `abs_pct` doesn't depend on predictions) by replicating `make_splits()`
and `eval_combo()`'s absence check on the original `windowed_k24.npy`: **14.238864%**,
matching the table to 6 decimal digits. Confirmed `raw.npy` is byte-identical (md5) to an
independent fresh re-run of the same refmap command, and the read-placement status counts
(EXACT/PLACED/MULTI/UNPLACED) match `nam_baseline_results.tsv`'s `n_placed`/`n_unplaced`
exactly.

## Step 2: the test split is not random — and it's the same two chromosomes for every founder

`crf/train_haploid.py`'s `make_splits()` does a **plain positional tail slice**
(`data[n_train+n_val:]`) of the windowed array, not a shuffle. The windowed array itself
(`ropebwt_npy_to_matrix.py`) is built by `bins_df.groupby("contig", sort=False)` — i.e.
contigs appear in **first-occurrence order in `bins.tsv`**, which turns out to be natural
chromosome order (chr1, chr2, …, chr10), with chr10 *last* among the 10 main chromosomes
(scaffolds don't reach the 512-row minimum to form even one window and are dropped
entirely). Reconstructing the exact window/split boundary for **all 24 founders** using
their original data:

**Every single founder's held-out test split lands on exactly `{chr10 (100%), chr9 (last
~35–40%)}`** — never any other chromosome. The reconstructed absent rate computed
independently within just those rows matches `nam_baseline_results.tsv`'s `abs_pct` column
**exactly** for all 24 founders (e.g. CML103 14.238864/14.238864, Tzi8 27.872930/27.872930,
CML52 8.571862/8.571862 — not approximately, to 6 significant figures). This fully explains
the mechanism: **`abs_pct`/`abs_acc`/`viterbi` are not a random genome-wide sample — they
are entirely determined by a founder's own placement behavior on chr10 plus the tail third
of chr9**, an artifact of an un-shuffled train/val/test split, not a deliberate design choice
to hold out those chromosomes.

## Step 3: CML103's chr10 is a real, dramatic, fleet-unique outlier

Computed each founder's own chr10-vs-other-chromosome self-absent ratio (bin-level, same
definition throughout):

| founder | chr10 absent% | other-chr absent% | **ratio** | viterbi |
|---|---:|---:|---:|---:|
| **CML103** | **18.54%** | **2.79%** | **6.65x** | 0.791 |
| Tzi8 | 34.21% | 18.36% | 1.86x | 0.521 |
| CML69 | 3.79% | 3.68% | 1.03x | 0.979 |
| *(all other 21 founders)* | — | — | **0.44x – 0.97x** | 0.87–0.99 |

**CML103 is a sharp, isolated outlier** — 6.65x is by far the largest ratio in the entire
24-founder fleet, more than 3.5x the next-highest (Tzi8's 1.86x, and Tzi8 is elevated
*everywhere*, 18.4% baseline, so its chr10 uplift is a smaller relative effect on an already
bad founder, not a localized anomaly). Every other founder shows chr10 to be at or *below*
their own average (ratios 0.44–1.03x) — chr10 is, if anything, the pangenome's easiest
chromosome for everyone except CML103.

## Step 4: the anomaly is localized within chr10, not chromosome-wide

Per-1Mb absent rate along CML103's own chr10 (152.5Mb total) shows the elevation
concentrated in a **~40Mb span, roughly 83–126Mb** (17% of 1Mb windows there exceed 50%
absent, several reaching 75–84%), against a normal ~3% baseline on the rest of the
chromosome (median window rate 3.03%). This is a real, spatially clustered structural
signature — not diffuse noise, and not a small single-locus blip either.

**Plausible mechanism (not confirmed)**: large structural variants segregating among the 26
NAM founder genomes relative to B73 are well documented — e.g. a >47Mbp reciprocal
translocation between chr9 and chr10 short arms in Oh7B (De novo assembly, annotation, and
comparative analysis of 26 diverse maize genomes, *Science* 2021). A comparably large,
CML103-specific structural divergence on chr10 (translocation, large indel, or knob/Ab10-like
heterochromatic expansion — Ab10 haplotypes are known to carry large extra chromatin blocks
on chr10's long arm) would produce exactly this signature: reads correctly generated from
CML103 fail to place back onto the B73-anchored pangenome coordinate across a large,
contiguous span. **No source found that specifically documents a CML103 chr10 SV** — this is
a testable hypothesis grounded in the observed pattern and documented NAM-founder precedent,
not an established fact; confirming it would require directly comparing the CML103 assembly's
chr10 against B73 (dotplot / SV caller), which is out of scope here.

## Bottom line

| question | answer |
|---|---|
| Is CML103's low accuracy explained by elevated genome-wide self-absent rate (like Tzi8/CML52)? | **No** — genome-wide rate is normal (~2.8–4.1%) |
| Then why is `abs_pct`=14.24% / viterbi=0.79? | The benchmark's test split is a **positional tail slice, not random**, and lands on `{chr10, chr9-tail}` for every founder |
| Why does that specifically hurt CML103? | CML103 has a **real, isolated, ~40Mb structural anomaly on chr10** (18.5% self-absent vs. 2.8% elsewhere, 6.65x — the largest such ratio in the fleet by a wide margin) that happens to sit exactly inside the fixed test window |
| Does this affect other founders' reported scores too? | The `{chr10,chr9-tail}` split is universal, but for the other 23 founders chr10 is unremarkable (ratios 0.44–1.03x), so their scores are roughly representative; CML103 is the one founder for whom this methodological quirk produces a substantially misleading number |

**CML103's poor fleet-accuracy ranking is a real, chr10-localized phenomenon, but the
magnitude reported by `nam_baseline_results.tsv` is inflated by a non-random test split that
happens to be dominated by exactly the one chromosome where CML103 diverges.** A shuffled or
chromosome-stratified train/val/test split (rather than a positional tail slice of a
contig-ordered array) would give a materially more representative accuracy number for
CML103 specifically, and is a worthwhile fix to `crf/train_haploid.py`'s `make_splits()`
for any future fleet-wide benchmarking, independent of the CML103 finding.

## Reproducing

All scripts and intermediates in the session scratchpad
(`step1_cml103_row_vs_bin.py`, `step2_test_split_bias.py` [superseded, had a contig-sort
bug — see step4], `step3b_selfcontained.py`, `step4_locate_test_split.py`,
`step5_chr10_fleet_check.py`, `step6_fleet_split_bias.py`) — not promoted into the repo. All
numeric reproductions used the **original, preserved** per-founder directories under
`scratchpad/nam_baseline/<Founder>/` (still present on disk for all 24 founders as of this
writing), not regenerated data, to guarantee exact agreement with
`results/nam_baseline_results.tsv`.
