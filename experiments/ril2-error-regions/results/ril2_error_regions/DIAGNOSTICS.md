# RIL2 error-region diagnostics: SSA occurrence, non-continuity, reads-to-intersection, anchor density

Three diagnostics requested by the user's colleague on top of the existing
N-gap/PAV-proxy/SSR annotation pass, all scored error-vs-background on the
same **210 real founder-path error intervals** (IDX-RIL2 Oh43xIl14H, 0.5x
coverage, `--bin-size=1`, corrected adaptive drop-index -- the most
rigorously validated error set from this line of work) against **1000
width-matched, chrom-length-weighted random background regions**. See
`/home/zrm22/.claude/plans/wondrous-discovering-octopus.md` for full design
rationale. All p-values are two-sided Mann-Whitney U (skewed distributions).
A fourth item (anchor density vs. error, testing the founder-touchdown-map
finding population-wide) was added afterward, same regions and convention.

Scripts: `scripts/region_occ_stats.py`, `scripts/region_noncontinuity.py`,
`scripts/reads_to_intersection.py`, `scripts/anchor_density_vs_error.py`.
Outputs (this directory): `region_occ_stats.json`,
`region_noncontinuity.json`, `reads_to_intersection.json`,
`anchor_density_vs_error.json`.

## 1. Pangenome occurrence-count ("SSA") variance

`ropebwt3`'s sampled suffix array (`.fmd.ssa`) itself holds no counts -- it
is a rank-to-genome-position lookup table for `locate()`. Occurrence
counts are FM-index interval sizes, obtained here by tiling B73 reference
sequence (150bp windows, 500bp stride, capped at 400 tiles/region) and
querying `ropebwt3 mem` directly against the `.fmd` -- a pure string
lookup, no reads, no refmap re-run. 121,352 tiles total; 568 (0.47%)
excluded for containing N (assembly-gap) bases, which otherwise produce a
degenerate, meaningless interval size in the hundreds of millions (found
via a background region that landed entirely in an N-gap: occ_mean
334,485,246 before the fix, vs. the next-highest real region ~15,000
after). 3 regions (1 error, 2 background) are fully inside an N-gap and
were dropped from this comparison entirely.

| statistic | error median (n=209) | background median (n=998) | p |
|---|---|---|---|
| occ_mean | 131.8 | 78.1 | 7.5e-4 |
| occ_cv (sd/mean) | 2.59 | 2.75 | 0.67 |
| frac_ge100 (tiles with occ>=100) | 0.100 | 0.075 | 2.1e-4 |

**Verdict:** error regions sit in measurably more repetitive pangenome
sequence -- both the mean occurrence and the fraction of highly-repeated
tiles are significantly elevated. The relative *spread* (CV) is not
different, so it's not that error regions are more erratic per unit of
their own repetitiveness -- they are just more repetitive overall. Cross-
check: `occ_mean` correlates weakly positively with the pre-existing
`ssr_fraction` annotation (Pearson r=0.11, Spearman r=0.24) -- weak, not
because of a bug, but because pangenome-wide dispersed repeats/paralogs
(what `occ` measures) and local tandem/microsatellite repeats (what SSR
measures) are different repeat classes that only partially overlap.

## 2. Non-continuity (fragmentation / repetitiveness / structural change)

Structural half reuses the existing Il14H-vs-B73 AnchorWave covered-
interval BED (`il14h_covered.merged.bed`) and the N-gap BED -- no new data.
Repetitiveness half reuses item 1's per-tile occurrence series (no new
query): the fraction of adjacent tiles whose occurrence jumps >2x
(`occ_jump_rate`), and the lag-1 autocorrelation of log10(occurrence)
along the region (`occ_log_acf1`).

| statistic | error median (n=210) | background median (n=1000) | p |
|---|---|---|---|
| covered_fraction (Il14H vs B73) | 0.420 | 0.731 | 1.3e-4 |
| n_breaks_per_10kb | 3.00 | 2.76 | 0.57 |
| occ_jump_rate | 0.453 | 0.396 | 8.6e-4 |
| occ_log_acf1 | 0.083 | 0.140 | 0.27 |

Sanity check passed: background `covered_fraction` mean = 0.6035, close to
the genome-wide Il14H covered fraction (1 - 42.4% PAV-proxy = 0.576).

**Verdict:** error regions carry substantially *less* total Il14H-vs-B73
alignment coverage (42% vs 73% of the region, median) -- a real, more
sensitive result than the earlier boolean "touches a PAV-proxy gap
somewhere" check, which showed almost no difference (87% vs 83%) because
nearly every region of any size touches *some* small gap. The breakpoint-
count-per-kb structural measure shows no difference, so it isn't that
error regions are more *fragmented* into many small aligned segments --
it's that a much larger *share* of the region itself is unaligned/PAV.
`occ_jump_rate` corroborates item 1's higher repetitiveness (choppier
occurrence signal in error regions), though `occ_log_acf1` doesn't reach
significance on its own.

## 3. Reads until the founder-support intersection collapses to 1

Uses the existing `raw.ps4g` for this exact run (bin-size=1, so
`refPosBinned` is real bp with almost no read-collapsing: 5,685,048 data
rows for 5,694,530 total reads). Per-read gameteSet is destroyed for EXACT
reads in refmap's own output, but PS4G preserves within-read co-occurrence
and only collapses *identical* (contig, bin, gameteSet) reads into one row
with a count -- expanding each row to `count` physical read-slots and
shuffling those (T=30 random orders/region, stable across seeds at this
scale) is mathematically exact, not an approximation of true per-read
order. Two variants: **strict** intersection (any single non-matching read
can break it), and **tolerant** (founders present in >=98% of reads seen
so far, after a 3-read minimum) since strict is fragile to one bad read.
`true_founder` for background regions comes from `simval_oracle_bed.
build_ril_mosaics`'s exact RNG-replayed truth mosaic; 2 background regions
spanning a true breakpoint have no single founder and are excluded from
the correctness stat. 61 background regions (0 error) had zero PS4G rows
in-window -- all narrow (down to 1bp) random windows with no read landing
inside by chance at 0.5x coverage, not a processing failure.

| statistic | error median (n) | background median (n) | p |
|---|---|---|---|
| strict reads-to-1 | 3.0 (183) | 6.0 (722) | 5.1e-9 |
| tolerant reads-to-1 | 4.0 (166) | 7.0 (702) | 2.6e-5 |
| strict frac_collapsed_to_empty (mean) | 0.40 | 0.19 | -- |

P(strict-converged error region converges to the **correct** founder) =
**0.169** (n=183). For comparison, spot-checked background regions
converge to the correct founder essentially every time they converge at
all (validates the mechanism is correct, not just a plausible-looking
number).

**Verdict:** this is the sharpest of the three findings, and it cuts
against a first guess. Error regions do **not** need more reads to
resolve -- they converge *faster* (median 3 reads vs 6). The problem is
that when they converge, they confidently converge on the **wrong**
founder 83% of the time. Error regions also collapse the strict
intersection to *nothing* (a genuinely incompatible pair of reads) roughly
twice as often as background (40% vs 19% of trials). Read together with
items 1-2: error regions sit in more repetitive, less Il14H-aligned
sequence where a *small* number of reads is enough to build spuriously
confident, wrong support for the sample's other true parent (or a
different founder) before the genuinely informative reads accumulate --
consistent with the session's earlier bp-distance-decay finding that
Oh43/Il14H read-support co-occurrence persists 10-50kb, far past normal LD
decay.

## 4. Anchor density vs. error: does poor founder-vs-B73 anchoring predict errors?

Follow-up test of the [[founder_touchdown_map]] finding (Il14H's assembly
fails to anchor to B73 specifically at chr5:200.6-203.5Mb, the site of a
real Oh43xIl14H decode error there): does that pattern hold **population-
wide**, or was chr5 an extreme anecdote? For each of the 210 error and 998
scoreable background regions (2 background regions skipped: span a true
breakpoint, no single true founder), look up the **true founder's** anchor
density (from `results/founder_touchdown_map/touchdown_matrix_100kb.json`)
at every 100kb bin the region overlaps, taking the min and mean across
overlapping bins. Two readings of the same data:

| statistic | error median | background median | p |
|---|---|---|---|
| raw anchor count (min in region) | 4.0 | 9.0 | 4.9e-7 |
| raw anchor count (mean in region) | 5.8 | 12.0 | 1.5e-9 |
| anomaly ratio -- count / cross-founder median at that bin (min) | 0.988 | 1.000 | 0.019 |
| anomaly ratio -- count / cross-founder median at that bin (mean) | 1.000 | 1.013 | 0.005 |

**Both readings are statistically significant, but they tell different-sized
stories.** Raw count: error regions' true founder has roughly HALF the
anchor density of background regions' true founder (highly significant,
large effect) -- the basic prediction holds. Anomaly ratio (which controls
for how hard the locus is for every founder, not just the true one): both
groups sit almost exactly at 1.0 (typical) -- a real but tiny effect. That
gap between the two readings means the raw-count effect is NOT mostly "the
true founder specifically anchors worse than its peers" (the chr5/Il14H
mechanism) -- it's mostly "error regions sit in loci that are harder to
anchor for everyone," consistent with item 1's separate finding that error
regions are more pangenome-repetitive in general.

**Classifying the 210 individual error regions** (ratio cutoff 0.5, count
cutoff = the background min-count median of 9) splits them roughly evenly
three ways **by region count** -- but not by bp, which is the metric the
headline founder-path error rate is actually weighted by:

| mechanism | n regions | % of regions | total bp | % of error bp | avg width |
|---|---|---|---|---|---|
| founder_specific_dropout (true founder specifically worse than its peers -- the chr5 pattern) | 67 | 31.9% | 11,863,657 (11.86 Mb) | **64.8%** | 177,070 bp |
| general_difficulty (locus hard for every founder, true founder not unusual) | 74 | 35.2% | 4,039,479 (4.04 Mb) | 22.1% | 54,588 bp |
| unexplained_by_anchor_density (anchoring is normal here; something else is wrong) | 69 | 32.9% | 2,390,964 (2.39 Mb) | 13.1% | 34,652 bp |
| **total** | 210 | 100% | **18,294,100 (18.29 Mb)** | 100% | -- |

(For context, the 998 scoreable background regions cover 88,667,346 bp /
88.67 Mb.)

**Verdict, revised**: by region *count* the three mechanisms look tied.
By bp -- the metric that actually determines the reported founder-path
error rate -- they are not: `founder_specific_dropout` regions run
3-5x wider on average than the other two categories, so despite being
under a third of regions by count, they account for **nearly two-thirds
of all error bp**. The chr5/Il14H pattern is therefore not a tied-for-
third contributor, it is the dominant driver of the RIL2 founder-path
error rate by bp-weighted impact, even though it is a minority of
distinct error *events*. The remaining third of error bp (general
difficulty + unexplained) is smaller in aggregate but still real, and the
`unexplained_by_anchor_density` bp in particular needs a different
explanation entirely (read co-occurrence/LD persistence, non-continuity,
or something not yet diagnosed).

## 5. Does the classification generalize beyond Oh43xIl14H 0.5x?

Item 4 was one sample (Oh43xIl14H, 0.5x, `binsize1`). Reran the same
extraction + classification (`scripts/multi_pair_anchor_generalization.py`)
across **all 5 manifest coverages x 5 IDX-RIL2 founder pairs** available
in the corpus (Oh43xIl14H, B73xCML103, B73xOh43, B97xCML103, Il14HxB97),
all using the `unfiltered-bin` tag uniformly since it's the only tag built
for the 4 non-Oh43xIl14H pairs -- note this means the Oh43xIl14H rows here
differ slightly from item 4's `binsize1` headline numbers (both valid, not
the same run). B73xCML103 and B73xOh43 have B73 as one parent; segments
truly labeled B73 are skipped (B73 is deliberately excluded from the
founder touchdown matrix as the reference itself), so those two pairs have
roughly half as many scoreable error regions as their raw error count.

| pair | coverage | n scoreable errors | dropout % of error bp | general % | unexplained % | p (Mann-Whitney, raw count) |
|---|---|---|---|---|---|---|
| Oh43xIl14H | 0.01x | 23 | 75.0% | 3.7% | 21.3% | 0.20 |
| Oh43xIl14H | 0.1x | 53 | 74.2% | 4.7% | 21.1% | 0.13 |
| Oh43xIl14H | 0.5x | 142 | 70.4% | 17.5% | 12.1% | 2.5e-6 |
| Oh43xIl14H | 1.0x | 243 | 61.4% | 20.6% | 18.0% | 0.0022 |
| Oh43xIl14H | 2.0x | 451 | 57.8% | 21.4% | 20.8% | 6.9e-9 |
| B73xOh43 | 0.1x | 9 | 91.2% | 0.0% | 8.8% | 0.44 |
| B73xOh43 | 0.5x | 11 | 95.9% | 3.5% | 0.6% | 0.11 |
| B73xOh43 | 1.0x | 21 | 99.2% | 0.4% | 0.5% | 0.0078 |
| B73xOh43 | 2.0x | 50 | 89.6% | 9.5% | 0.9% | 2.6e-4 |
| Il14HxB97 | 0.01x | 20 | 42.6% | 30.6% | 26.8% | 0.87 |
| Il14HxB97 | 0.1x | 60 | 69.7% | 8.7% | 21.6% | 0.032 |
| Il14HxB97 | 0.5x | 158 | 72.4% | 6.5% | 21.1% | 0.0062 |
| Il14HxB97 | 1.0x | 267 | 67.7% | 7.5% | 24.8% | 0.038 |
| Il14HxB97 | 2.0x | 484 | 60.8% | 16.3% | 22.9% | 1.4e-9 |
| B97xCML103 | 0.01x | 22 | 38.3% | 46.2% | 15.5% | 0.25 |
| B97xCML103 | 0.1x | 51 | 64.7% | 22.3% | 13.0% | 0.29 |
| B97xCML103 | 0.5x | 96 | 53.0% | 32.5% | 14.6% | 0.074 |
| B97xCML103 | 1.0x | 197 | 55.6% | 27.4% | 17.0% | 0.0069 |
| B97xCML103 | 2.0x | 357 | 52.7% | 32.4% | 14.9% | 3.0e-8 |
| B73xCML103 | 0.01x | 12 | 0.0% | 1.7% | 98.3% | 0.13 |
| B73xCML103 | 0.1x | 25 | 7.6% | 0.0% | 92.3% | 3.1e-4 |
| B73xCML103 | 0.5x | 84 | 12.2% | 2.5% | 85.3% | 5.0e-18 |
| B73xCML103 | 1.0x | 132 | 10.5% | 4.9% | 84.6% | 1.8e-29 |
| B73xCML103 | 2.0x | 236 | 20.4% | 6.6% | 72.9% | 9.5e-32 |

**Verdict: mixed, and the split is informative rather than noise.**

- **Within Oh43xIl14H, the finding holds at every coverage** -- dropout
  stays the majority contributor (58-75% of error bp) across all 5
  coverages, though `general_difficulty` grows and `dropout` shrinks
  somewhat as coverage rises.
- **It generalizes strongly to any pair involving Oh43 or Il14H**:
  B73xOh43 (89-99%, even stronger than Oh43xIl14H itself) and Il14HxB97
  (43-72%) both show founder-specific dropout as the dominant mechanism at
  every coverage tested. B97xCML103 is a weaker, mixed case (dropout and
  general_difficulty roughly co-dominant, 38-65% vs 22-46%).
- **It does NOT generalize to B73xCML103** -- dropout explains only
  0-20% of error bp there, and `unexplained_by_anchor_density` dominates
  overwhelmingly (73-98%) at every coverage, with very strong significance
  (p as low as 1e-32 on the raw-count test, meaning the *direction* of the
  effect is real, just not explained by this locus-anchor mechanism).
  Something else entirely drives B73xCML103's errors, and it is not this
  founder-touchdown-map story.

Net: the chr5/Il14H mechanism is a real, generalizable phenomenon tied to
specific founder identities (Oh43 and Il14H's assemblies both show this
locus-specific anchor-dropout pattern broadly, not just at one chr5
locus) -- but it is not universal across all founder pairs, and
B73xCML103 needs its own separate investigation rather than assuming this
mechanism applies.

## 6. Why is B73xCML103 unexplained? It isn't a locus problem -- it's B73/CML103 mutual confusion

Dug into the actual decoded calls for B73xCML103's wrong intervals (all 5
coverages, `unfiltered-bin`, `scripts/extract_error_regions.py`'s
`extract_wrong_intervals` reused directly, no new script needed for this
part):

| coverage | n wrong | CML103-true decoded as B73 | B73-true decoded as CML103 | mutual confusion, % of all wrong bp |
|---|---|---|---|---|
| 0.01x | 22 | 12/12 (100%) | 10/10 (100%) | 100.0% |
| 0.1x | 47 | 24/25 (96.0%) | 19/22 (86.4%) | 92.7% |
| 0.5x | 135 | 80/84 (95.2%) | 40/51 (78.4%) | 88.9% |
| 1.0x | 209 | 122/132 (92.4%) | 61/77 (79.2%) | 89.6% |
| 2.0x | 341 | 197/236 (83.5%) | 80/105 (76.2%) | 79.9% |

**B73xCML103's errors are, at every coverage, almost entirely (80-100% of
wrong bp) the model swapping B73 and CML103 for each other** -- not
scattered toward other founders, and not concentrated at any particular
kind of locus. This is a fundamentally different failure mode than
anything the founder-touchdown-map can see: that tool asks "does this
non-reference founder's own assembly anchor well to B73," which has no
way to represent "how easily does the model confuse this founder *with*
B73" (B73 is deliberately excluded as a comparable row -- it's the
reference axis, not a founder being compared against it). That's exactly
why item 5 classified these as `unexplained_by_anchor_density`: the tool
wasn't wrong, it just cannot see this class of error by construction. The
confusion is also directional -- CML103-true->B73 is consistently more
common than the reverse at every coverage -- consistent with some kind of
bias toward defaulting to the reference genome under ambiguous read
support, not a purely symmetric mix-up.

**Tested hypothesis: is CML103 unusually similar to B73 overall (making
B73xCML103 uniquely hard to tell apart), unlike Oh43 (which shows genuine
locus-specific dropout instead of reference confusion)?** Measured
directly via `scripts/founder_divergence_from_b73.py` -- SNP density per
covered bp from each founder's own AnchorWave gVCF vs. B73, genome-wide:

| founder | SNP density (per kb covered) |
|---|---|
| CML103 | 18.808 |
| Oh43 | 16.528 |

**Hypothesis falsified.** CML103 is *more* divergent from B73 than Oh43
is genome-wide, not less -- the opposite of what would explain the
mutual-confusion pattern via overall sequence identity. The mutual-
confusion finding itself is solid and robust across all 5 coverages; what
causes it specifically for CML103 (and not Oh43) was still open at this
point -- a patchy, LOCAL mechanism (shared IBD blocks or introgressed
segments with B73 at specific loci, echoing this session's separate
finding that Oh43/Il14H read-support co-occurrence persists far past
normal LD decay at some loci) looked more consistent with "high
genome-wide divergence but still frequently confused" than any
genome-wide identity effect -- tested directly below.

**Tested directly and confirmed.** Generalized `scripts/bp_distance_decay.py`
(originally built for Oh43/Il14H) to take `--individual/--parent-a/
--parent-b/--tag` as CLI args and reran it for B73/CML103 (`unfiltered-bin`,
all 5 coverages). Compared bin-by-bin against the already-established
Oh43/Il14H curve:

| bp distance | B73/CML103 P(j&#124;i) | Oh43/Il14H P(j&#124;i) | n_pairs (0.5x) |
|---|---|---|---|
| 0-150 | 0.739 | 0.706 | 997K |
| 1k-1.5k | 0.725 | 0.712 | 6.1M |
| 10k-20k | 0.667 | 0.666 | 110M |
| 20k-50k | 0.641 | 0.642 | 323M |
| 50k-100k | 0.614 | 0.614 | 519M |
| 100k-500k | 0.542 | 0.538 | 343M |

**The two curves are essentially identical, bin for bin, across the
entire well-sampled range (0 to 100-500kb), at every coverage checked.**
B73/CML103 shows the exact same signature as Oh43/Il14H: read-support
co-occurrence staying roughly double background (base rate ~0.37-0.40)
far past where ordinary LD should have decayed (~1-2kb), persisting out
to hundreds of kb. (The one bin that diverges, 500k-2M, has a much
smaller and less reliable sample there for B73/CML103 -- ~790K pairs vs.
343M in its neighboring bin -- consistent with the script's documented
row-offset cap reaching that far only in unusually sparse genomic
stretches at higher row density; not read as a real difference.)

**Verdict (part 1):** persistent, LD-decay-defying shared read support is
real and confirmed -- B73/CML103 shows it at essentially the same
strength Oh43/Il14H did.

**Discriminating test, and a correction: does this actually explain WHY
CML103 (mutual confusion) differs from Oh43 (genuine anchor dropout)?**
Ran the same decay analysis for B73xOh43 -- the pair item 5 found to be
the *strongest* founder-specific-dropout case (89-99% of error bp), not a
confusion case -- expecting it to show a much WEAKER co-support curve than
B73xCML103 if persistent co-support were the thing that specifically
causes reference confusion.

**It did not turn out that way.** B73xOh43 shows the *strongest*
persistence of all three pairs, not the weakest:

| bp distance | B73xOh43 P(j&#124;i) | B73xCML103 | Oh43xIl14H | (0.5x) |
|---|---|---|---|---|
| 10k-20k | 0.714 | 0.667 | 0.666 | |
| 20k-50k | 0.696 | 0.641 | 0.642 | |
| 50k-100k | 0.687 | 0.614 | 0.614 | |
| 100k-500k | 0.617 | 0.542 | 0.538 | |
| base_rate | 0.424 | 0.377 | 0.368 | |

B73xOh43's curve sits above both other pairs at every distance, and its
base rate is the highest of the three -- the opposite of the predicted
discriminating pattern.

**Verdict (part 2, corrected):** persistent co-support is confirmed as a
real, general phenomenon across this founder panel (now shown in three
different pairs) -- but it does NOT discriminate between the two failure
modes. It is elevated for both the founder-specific-dropout pair
(B73xOh43, most elevated of the three) and the mutual-confusion pair
(B73xCML103) alike. So it looks like background relatedness/IBD structure
common to the whole panel, not the specific variable that determines
*which* error mechanism (dropout vs. reference confusion) a given pair
falls into. What makes CML103 specifically prone to being confused *with
B73 itself*, rather than showing dropout the way Oh43 does, remains
genuinely open -- the persistent-co-support hypothesis correctly
identified a real phenomenon but is not the answer to that specific
question.

## Regenerating dropped large intermediates

`il14h_covered.bed`/`.sorted.bed`/`.vsorted.bed` (1.1GB each) and
`il14h_covered_raw.tsv` (926MB) were not moved into this directory (they
only exist to produce the kept `il14h_covered.merged.bed`). To rebuild:

```
bcftools query -f '%CHROM\t%POS0\t%END\n' data/maize_v2_rebuild/gvcf_sorted/Il14H.g.vcf.gz \
  | awk '$1 ~ /^chr[0-9]+$/' > il14h_covered_raw.tsv
sort -k1,1V -k2,2n il14h_covered_raw.tsv > il14h_covered.vsorted.bed
bedtools merge -i il14h_covered.vsorted.bed > il14h_covered.merged.bed
bedtools complement -i il14h_covered.merged.bed -g genome.txt > il14h_pav_proxy.bed
```
