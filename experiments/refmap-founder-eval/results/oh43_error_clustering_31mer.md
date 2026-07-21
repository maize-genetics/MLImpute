# Are Oh43-absent bins spatially clustered / driven by recurrent 31-mer repeats?

Follow-up to `oh43_residual_4pct.md` (four residual mechanisms, dominated by repeat/paralog
mismapping) and `repetitiveness_and_copy_number.md` (31-mer *occurrence-count* correlates
with absence: median 311 vs. 55, monotonic 3.4%→11.4% gradient across repetitiveness
deciles). Those established a **correlation** with local repetitiveness but never asked
whether error bins concentrate in the *same* recurrent 31-mer sequences, whether the same
bins error across the whole 25-founder fleet, or checked clustering rigorously rather than by
eye. This closes those four gaps.

**Signal**: Oh43-absent bins (≥1 EXACT/PLACED read positively excludes Oh43 from a
256bp bin's credited founder set), the same definition used throughout the Oh43
investigation. **Scope**: diagnose-only; all scripts/intermediates in the session
scratchpad, this is the only promoted artifact.

## Reproduction gate

Regenerated the canonical untrimmed Oh43 refmap output from scratch (patched
64-carrier-cap binary, `--ref-prefix=B73 --max-occ=-1 --lift ... -t 20`, 38.5s) and rebuilt
the per-bin absent table. **33,793 / 668,108 bins absent = 5.0580%**, and **main-chr vs.
scaffold split 4.57% vs. 63.81%** — both reproduce the documented figures exactly before any
downstream analysis was trusted.

## Check A — 31-mer recurrence: yes, errors concentrate in a small set of repeat sequences

For every Oh43-absent bin on the main chromosomes (n=30,250) and a size-matched random
sample of present bins, tiled the B73 sequence (256bp + 15bp flanks) into canonical
(strand-collapsed) 31-mers and counted, for each distinct 31-mer, how many *distinct bins*
it appears in ("recurrence").

| | absent bins | present-bin control |
|---|---:|---:|
| mean recurrence | 1.354 | 1.212 |
| kmers recurring in ≥5 bins | 2.24% | 1.37% |
| max recurrence (bins) | 607 | 123 |
| bins containing a "hub" kmer (recurs ≥20 bins) | 32.3% | 21.5% |

Mann-Whitney U comparing the two recurrence distributions: p≈0 (absent > present,
overwhelming). The absent-bin distribution has a **much heavier tail** — its single most
recurrent 31-mer appears in 607 distinct absent bins (5x the present-control's max of 123).
**This is the literal answer to the original question: yes, Oh43-absent bins are not just
independently repetitive — a relatively small number of specific 31-mer sequences recur
across hundreds of distinct absent bins each.**

The top-30 recurrent 31-mers were queried for uncapped pangenome copy number
(`ropebwt3 fa2kmer` design reused via direct `ropebwt3 suffix` against the full 25-founder
`.fmd`): **1.1–1.9 million occurrences pangenome-wide**, an order of magnitude above the
most-repetitive decile (58,964 mean) reported in `repetitiveness_and_copy_number.md` —
these are extreme, satellite-class repeats, not merely "somewhat repetitive" sequence.

## Check D — naming the repeat family: knob180 satellite (plus two other distinct causes)

Built a small BLAST reference library from **9 real GenBank records fetched live via NCBI
eutils** (`curl`, not paraphrased through a summarizing tool, to avoid corrupting exact DNA
sequence): four knob180 clones (`DQ352544.1` "MK180", `AF030934.1`, `AF030937.1`,
`AF030940.1`), the TR-1 knob repeat (`DQ186871.1`), and four CentC centromeric-satellite
clones (`HN174148.1`, `HN174144.1`, `HN174141.1`, `CL569242.1`).

- **All 30 of the top recurrent 31-mers hit knob180 at 100% identity** (16→`DQ352544.1`,
  12→`AF030937.1`, 2→`AF030934.1`). The genome-wide 31-mer-recurrence signal in Check A is
  unambiguously the **knob180 heterochromatic satellite repeat** — its extreme, multi-million
  pangenome copy number explains both the recurrence tail and the extreme repetitiveness
  deciles previously reported.
- Targeted BLAST of individual absent-bin sequences from the four most extreme 1Mb hotspot
  windows (Check C) showed the mechanism is **not uniform**:
  - `chr6:21–23Mb` (the classic cytological knob on 6S) and `chr4:2Mb`: no hit in this small
    library — plausibly a divergent knob-repeat variant or interspersed knob-associated
    retroelement not covered here; inconclusive without a fuller TE database.
  - `chr7:57Mb` (new hotspot, 46.9% absent, n=98 with n≥20 filter): clean, strong hits to
    **CentC** (up to 98.7% identity, e=4e-35) — a **centromeric satellite**, a genuinely
    distinct repeat family from knob180.
  - `chr2:196Mb` (new hotspot): sequence extraction returned pure `N` runs; the full
    chr2:196–197Mb B73 window is **36.1% N-content** — this "hotspot" is not a repeat effect
    at all, it's a **B73 assembly gap**, echoing (at far larger scale) the small
    assembly-coverage-gap mechanism already noted in `oh43_residual_4pct.md`.

**Bottom line: knob180 is the dominant single repeat family, but hotspots are not all the
same mechanism** — centromeric satellite and outright assembly gaps also contribute.

## Check C — rigorous spatial clustering: real but scale-dependent

Using the ordered per-contig bin sequence (main chr1–10 only) and a within-contig
permutation null (1000 perms, positions fixed, labels shuffled):

- **Immediate-neighbor (256bp) adjacency**: 393 observed adjacent absent-absent bin pairs
  vs. null mean 142.4 (sd 12.3) — **z=20.3, p<0.001, 2.76x enrichment**. Absent bins are
  significantly more likely to sit right next to another absent bin than chance predicts.
- **Distance-decay confirms a real, local effect that fades with scale**:

  | threshold | enrichment vs. null | z | p |
  |---:|---:|---:|---:|
  | 256bp (1 bin) | 2.64x | 18.9 | <0.002 |
  | 512bp (2 bins) | 2.31x | 21.6 | <0.002 |
  | 1.28kb (5 bins) | 1.94x | 26.0 | <0.002 |
  | 5.12kb (20 bins) | 1.35x | 21.7 | <0.002 |
  | 25.6kb (100 bins) | 1.05x | 9.4 | <0.002 |

  All highly significant, but the enrichment **decays smoothly from 2.6x at 256bp toward 1x
  by ~25kb** — a classic signature of clustering driven by repeat elements that span several
  adjacent bins each (consistent with knob/satellite arrays), not genome-wide long-range
  structure.
- **Mean nearest-neighbor distance genome-wide** (34.2kb observed vs. 34.2kb null, z=-0.05,
  n.s.) shows **no detectable clustering at the whole-chromosome scale** — the local effect
  above is real but too small a fraction of all absent-bin pairs (only ~2.5% of absent bins
  sit in a run of ≥2 adjacent bins) to move a genome-wide average. Both results are correct
  and consistent: real, significant local clustering; no detectable large-scale clustering.
- The **known hotspots reproduce exactly** on a fresh per-1Mb Manhattan track (chr5:200–202Mb,
  chr6:4–5Mb & 21–23Mb, chr8:160–161Mb, chr9:0–1Mb) and several **new, even more extreme**
  ones are visible at ≥20-bin-covered resolution: **chr6:21–22Mb reaches 92.9% absent**
  (n=156), chr8:160Mb 65.9% (n=123), chr9:0Mb 64.7% (n=116), chr7:57Mb 46.9% (n=98), chr2:196Mb
  48.7% (n=187). (Two visually striking spikes at chr1:~170Mb and chr9's far end were checked
  and are **low-n artifacts** — n=8–16 bins with data — not statistically robust hotspots;
  filtering to n≥20 removes them cleanly.)

## Check B — cross-founder repeatability: a real, secondary mechanism (not the majority)

Reused persisted per-founder refmap output already on disk
(`scratch/nam_trim_sweep/q35_l75/<Founder>/`, 24 founders with reads; M162W/Tx303 excluded
as in the fleet baseline) to compute each founder's own self-absent bins on the shared B73
bin grid, then tallied how many founders are self-absent at each bin.

**Sanity check (and a bonus finding)**: 22 of 24 founders cluster tightly at 1.9–5.6%
self-absent, matching Oh43's 2.87% (untrimmed) / 2.87%-ballpark. **Two sharp outliers:
CML52 (11.3%) and Tzi8 (19.9%)** — exactly the two lowest-accuracy founders flagged in
`nam_baseline.md` (Tzi8 viterbi=0.52, CML52=0.89). This directly explains their poor fleet
accuracy via elevated placement-absence, a new corroborating result. **CML103 (viterbi=0.79,
the other flagged low-accuracy founder, previously unresolved) shows a perfectly normal
4.09% self-absent rate** — its accuracy problem is *not* explained by this mechanism, which
sharpens rather than resolves that open question.

**Overlap with Oh43-absent bins**:

| | mean n_founders_absent (of 24) | mean frac_founders_absent |
|---|---:|---:|
| Oh43-absent bins (n=33,793) | 0.38 | 12.75% |
| Oh43-present bins (n=634,315) | 0.07 | 3.07% |

~4.2x enrichment. At stricter thresholds: 14.5% of Oh43-absent bins have ≥50% of the other
24 founders also self-absent there (vs. 3.1% baseline, 4.7x), and 7.6% have ≥75% (vs. 1.1%,
6.7x) — real, statistically clear evidence for sample-independent "repeatable hard regions."
**But it is not the dominant pattern for individual bins**: 92.2% of Oh43-absent bins still
have ≤1 other founder self-absent there. Cross-founder repeatability is a genuine,
non-trivial contributing mechanism, not the majority explanation for any single Oh43-absent
bin.

## Bottom line

| question | answer |
|---|---|
| Do errors cluster in specific recurrent 31-mer sequences? | **Yes** — heavy-tailed recurrence (max 607 vs. 123 bins, p≈0), driven almost entirely by one repeat family |
| What is that family? | **knob180 satellite** (100% identity, all top-30 kmers) — pangenome copy number 1.1–1.9M; but hotspots are not all knob180 (CentC centromeric satellite and a bona fide B73 assembly gap also found) |
| Are error bins spatially clustered? | **Yes at fine scale** (2.6x enrichment at 256bp, decaying to ~1x by 25kb, all p<0.002) but **no detectable clustering genome-wide** (mean NN distance indistinguishable from null) — small, real, local effect |
| Do the same bins error across many founders? | **Partially** — 4-7x enrichment for cross-founder co-absence at Oh43-absent bins, but 92% of individual Oh43-absent bins are still ≤1-other-founder "unique" |
| Bonus | CML52/Tzi8's known low fleet accuracy is explained by elevated self-absent rate; CML103's is not (still unresolved) |

Taken together: Oh43's residual absent-bin signal is **not diffuse noise** — it has a real,
identifiable repeat-family signature (knob180 dominant, with centromeric satellite and
assembly gaps as secondary distinct causes) and measurable, if fine-scale, spatial structure.
It is, however, a **composite** of several mechanisms rather than one clean cause, consistent
with (and now sharpening) the multi-mechanism picture in `oh43_residual_4pct.md`.

## Reproducing

All scripts and intermediates live in the session scratchpad (`step0_build_bins.py`,
`step1_check_c_clustering.py`, `step1b_nn_threshold.py`, `step2_check_a_kmer.py`,
`step3_check_b_cross_founder.py`, plus the BLAST reference library
`maize_repeat_refs.fa`/`maize_repeat_db*` and result tables `check_{a,b,c,d}_*.tsv`,
including the per-1Mb Manhattan plot `check_c_clustering.png`) — not promoted into the repo,
per this project's diagnose-in-scratch-first pattern. The reference FASTA was fetched live
from NCBI via direct `curl` (not through a summarizing fetch tool) to avoid corrupting exact
DNA sequence content; accessions are cited inline above for anyone wanting to re-fetch them.
