# Sequencing error vs. contamination: a controlled test of the residual ~5% Oh43-absent rate

Follow-up to `oh43_residual_4pct.md`, which decomposed the residual Oh43-absent rate
**correlationally** (absent reads skew low-quality, credited "other founder" set is diffuse).
This investigation replaces that correlational evidence with a **controlled, causal** test of
three predictions:

1. **P1** — error-free 151bp reads sampled from `Oh43.fa` will not mismap.
2. **P2** — the same reads with 1–2 injected substitutions will mismap at an elevated rate;
   an ideal tool would fail to place them (UNPLACED) rather than mis-credit another founder.
3. **P3** (alternative hypothesis) — the *real* reads that miss Oh43 are enriched for the
   affinity of a specific contaminating line (barcode-mismatch contamination), rather than
   diffusely spread across founders (repeat/paralog confusion).

**Scope**: whole-read mode (no `--kmer`), the patched 64-carrier-cap `ropebwt3-phg refmap`
binary (`ropebwt3-phg/.claude/worktrees/refmap-ps4g-numpy/ropebwt3`), same index used
throughout this investigation. Diagnose-only — no code or repo changes. All read sets are
freshly simulated from `fastas/Oh43.fa` / `fastas/B97.fa` (P1/P2 and the P3 positive control);
P3's primary test uses the real, downsampled wet-lab reads in `bench_ps4g_npy/reads/Oh43_1M.fastq`.

## Reproduction gate

Before trusting any new number, the canonical command was re-run from scratch on the real
`Oh43_1M.fastq` with the current binary:

```
ropebwt3 refmap --ref-prefix=B73 --max-occ=-1 --lift=<lift> -t 20 \
  --ps4g=... --npy=... <fmd> Oh43_1M.fastq
```

Status mix (EXACT 301,796 / PLACED 411,931 / MULTI 265,438 / UNPLACED 20,835) and bin-level
Oh43-absent rate (**33,793 / 668,108 = 5.0580%**) reproduce the documented 5.06% figure exactly
— confirming the right binary/parameters before proceeding.

## Method: read simulation

`N=199,742` 151bp windows were sampled from the 10 main Oh43 chromosomes (chr1–chr10; 258 of
200,000 draws dropped for containing `N`), via a single batched `samtools faidx -r` call. The
**same** sampled positions were reused across four read sets so mutation load is the only
variable:

- `sim_m0` — clean (verbatim Oh43 sequence).
- `sim_m1` / `sim_m2` — 1 or 2 random substitutions at random positions.
- `sim_realistic` — per-base substitution with probability `10^(-Q/10)`, using **real quality
  strings reservoir-sampled from `Oh43_1M.fastq`**, carried into the emitted read — this
  anchors the injected error spectrum to the empirical data.

All sets were mapped with the identical canonical command and scored by the same bin-level
Oh43-absent metric used in the reproduction gate (computed from `--npy`, not the per-read TSV,
because **EXACT-status reads never list their carrier set in the TSV** — only in the npy — so
the TSV alone would systematically undercount EXACT-mechanism exclusions).

## Results: P1 and P2

| set | reads mapped (EXACT+PLACED bins) | bin-level Oh43-absent | Δ vs clean | PLACED reads: %credit Oh43 | %mis-credit (bad) |
|---|---:|---:|---:|---:|---:|
| `sim_m0` (clean) | 160,702 | **0.0000%** | — | 100.00% | 0.00% |
| `sim_m1` (+1 sub) | 158,348 | **9.7008%** | +9.70pp | 85.66% | 14.34% |
| `sim_m2` (+2 subs) | 154,943 | **16.4009%** | +16.40pp | 76.24% | 23.76% |
| `sim_realistic` (real error model) | 156,943 | **4.1435%** | +4.14pp | 94.13% | 5.87% |
| *(empirical, real reads, all bins)* | 668,108 | *5.0580%* | — | — | — |

Status mix shifted from clean to mutated sets exactly as expected: EXACT count fell
(67,487→66,442→64,767) as mutations knocked reads off a verbatim B73 match, and UNPLACED count
rose only modestly (603→911→1,079 for m0→m1→m2) — the tool overwhelmingly responds to injected
error by **mis-crediting a founder (PLACED, wrong)**, not by declining to call (UNPLACED). The
`sim_realistic` set shows a larger UNPLACED jump (603→3,455) because its real-quality-driven
model occasionally injects several errors on the worst-quality reads, which do get correctly
rejected — but even so, 5.87% of its PLACED reads still mis-credit a wrong founder.

**P1 — confirmed, decisively.** 199,742 error-free reads sampled from Oh43's own main
chromosomes produced **zero** Oh43-absent bins. There is no detectable repeat/paralog floor at
this sampling density on the main chromosomes: every clean read that RefMap placed confidently
(EXACT or PLACED) credited Oh43. (The residual doc's repeat/paralog mechanism is not
contradicted — it concentrates on unplaced scaffolds and specific repeat hotspots, which are
a small fraction of genome-wide bases and were under-sampled at N=200k; this result shows the
*sequencing-error-free* floor genome-wide is effectively zero, not that repeats never cause
mismapping anywhere.)

**P2 — confirmed.** The absent rate rises monotonically and steeply with mutation count:
0% → 4.14% (realistic) → 9.70% (1 sub) → 16.40% (2 subs). Critically, the *mechanism* of loss
is exactly what a "perfect vs. imperfect tool" framing predicts should be watched: an ideal
tool would push all lost reads to UNPLACED (declines to assert); this one instead mis-credits
a wrong founder for the large majority of lost reads (85.7–76.2% of PLACED reads still "confidently"
call a founder even after 1–2 injected errors, and it's the wrong one for 14–24% of them). This
is the exact brittleness the residual doc flagged (exact/near-exact FM-index matching is
sensitive to single-base errors in a way gapped aligners are not).

**Quantitative tie-in.** `sim_realistic` — whose per-base error model is anchored to real
Phred qualities sampled from the actual Oh43_1M reads — lands at **4.14%**, within ~0.9
percentage points of the empirical **5.06%**, despite sampling only the main chromosomes (the
empirical figure includes unplaced scaffolds, which the residual doc found are ~64% absent
locally and contribute disproportionately). This is strong, direct, causal evidence that
**sequencing error alone, layered on top of the (near-zero) clean-read floor, is sufficient to
explain essentially the entire residual ~5% Oh43-absent rate** — no contamination is required.

## Results: P3 (contamination)

### Real-read signal (refmap-native)

Among the real `Oh43_1M.fastq` reads, 38,117 positively exclude Oh43 (28,292 PLACED-excluding
+ 9,825 EXACT-excluding — the EXACT count matches the residual doc's figure exactly, confirming
methodology; the PLACED count is somewhat higher, likely reflecting this fresh run vs. the
original benchmark snapshot). Tallying which founders **are** credited across the 28,292
PLACED-absent reads' carrier sets:

| founder | share of absent reads | founder | share of absent reads |
|---|---:|---|---:|
| NC350 (top) | 18.80% | CML322 | 15.88% |
| Ki11 | 18.10% | CML247 | 15.86% |
| Mo18W | 17.11% | CML277 | 15.63% |
| CML333 | 16.99% | M162W | 15.53% |
| Ki3 | 16.85% | ... | ... |
| CML228 | 16.60% | Oh7B (lowest) | 12.47% |

**Diffuse — no dominant founder.** All 23 non-Oh43 founders cluster tightly between 12.5% and
18.8%, a spread of only ~6 points top-to-bottom with no single line standing out. This is the
repeat/paralog signature: a shared, ambiguous locus recruits many phylogenetically similar
founders roughly equally, not one specific contaminating line.

### Positive control: B97 spike-in

19,970 clean 151bp reads were simulated from `fastas/B97.fa` the same way as the Oh43 sets, and
mapped with the identical canonical command — modeling what genuine cross-line contamination
would look like in this pipeline. Result: **59.04% of B97 reads are flagged Oh43-absent**
(vs. ~0–5% for genuine/simulated Oh43 reads) — contamination, if present, would be easy to
detect at the aggregate absent-rate level alone. More importantly, among the B97 reads that
lose Oh43 credit, the credited-founder distribution is:

| founder | share of B97-spike absent reads |
|---|---:|
| **B97** | **100.00%** |
| Ms71 (next highest) | 27.38% |
| Oh7B | 23.57% |
| ... (remaining 21 founders) | 16.8–21.4% |

A **sharp, coherent, deterministic spike at the true contaminant** (100%), cleanly separated
from the next-highest founder (27.4%) and the rest of the diffuse background (~17–21%, the same
phylogenetic-similarity floor seen in the real data above). This confirms the test has power:
if a real contaminant were present in the Oh43 pool, its founder would stand out clearly above
this diffuse floor.

### Independent corroboration: 25-founder minimap2 sweep

Bypassing RefMap's FM-index placement entirely, a subsample of 2,500 real absent reads and
2,500 matched non-absent reads were aligned independently with `minimap2 -x sr` against each of
the 25 founder FASTAs, and each read's best-identity founder(s) tallied (restricted to
effectively-unambiguous best hits, n_tied&le;2 — 317/1,946 absent and 235/2,459 baseline reads
qualified; most 151bp reads tie across many founders at this scale, consistent with the
residual doc's repetitiveness findings):

| founder | share of absent reads | share of baseline (non-absent) reads |
|---|---:|---:|
| Il14H (top) | 12.62% | 14.89% |
| P39 | 8.20% | 5.96% |
| HP301 | 7.57% | 5.96% |
| CML333 | 7.26% | 4.26% |
| M162W | 7.26% | 4.68% |
| ... (19 more founders) | 3.5–7.0% | 1.7–5.5% |
| **Oh43** | **3.79%** | **37.87%** |

Two things confirm the method is working correctly and corroborate the refmap-native result:
(1) **Oh43 dominates the non-absent baseline** (37.87% best-hit share — by far the largest of
any founder, exactly as expected since these are genuinely Oh43-derived reads that mapped
correctly) **but is depleted among the absent-defining reads** (3.79%, near the bottom) — this
is definitionally consistent, since absent reads were selected specifically because RefMap
didn't credit Oh43. (2) Among the absent reads, no non-Oh43 founder rises above 12.6% — the
same diffuse, no-standout-line pattern as the refmap-native tally, independently confirming it
via a completely different (gapped, mismatch-tolerant) aligner.

### Verdict

**P3 — rejected.** The real Oh43-absent reads show the diffuse signature by two independent
methods (refmap-native: max 18.8%, ~6-point spread; minimap2 25-founder sweep: max 12.6%,
~9-point spread), not the coherent single-line spike (100%, ~73-point separation from the
runner-up) that the B97 positive control demonstrates contamination *would* produce. No
candidate contaminant line clears the diffuse background under either method. Barcode-mismatch
contamination is not supported by this data; the residual absent rate is adequately explained
by sequencing error (P1/P2 above) and genuine repeat/paralog structure (as already established
in `oh43_residual_4pct.md`).

## Bottom line

| prediction | verdict | key evidence |
|---|---|---|
| P1: clean reads don't mismap | **Confirmed** | 199,742 error-free Oh43 reads → 0.00% absent |
| P2: injected errors elevate mismap, mostly via mis-crediting not declining | **Confirmed** | 0%→4.14%→9.70%→16.40% dose-response; 14–24% of PLACED-lost reads mis-credit a wrong founder rather than going UNPLACED |
| P3: contamination from a specific line | **Rejected** | real absent reads diffuse (12.5–18.8% across all founders) vs. B97 positive control's coherent 100% spike |

**Sequencing error, not contamination, explains the residual ~5% Oh43-absent rate.** The
`sim_realistic` set (error model anchored to real base qualities) reproduces the empirical rate
to within ~1 percentage point using only the main chromosomes, and the contamination alternative
is directly falsified by contrast with a calibrated positive control.

## Reproducing

All scripts and intermediates live in the session scratchpad (`sim_reads.py`,
`map_and_score.py`, `pred3_identify_absent.py`, `pred3_minimap_analysis.py`,
`extract_fastq_subset.py`, `bin_absent_rate.py`) — not promoted into the repo, per the
diagnose-only scope. Fresh refmap output (npy/ps4g/tsv) for the real-read reproduction gate and
P3 analysis was regenerated from scratch rather than reusing `bench_ps4g_npy/results_fixed/`,
which was found to predate the carrier-cap fix (it reproduces the old 8-cap ~11% rate, not the
64-cap ~5% rate) despite its name.
