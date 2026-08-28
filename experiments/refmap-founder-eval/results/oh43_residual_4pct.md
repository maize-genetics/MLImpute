# Decomposing the residual ~4-5% Oh43-absent rate

Follow-up to `oh43_absent_rootcause.md`, which fixed the 8-carrier reporting cap
(10.85% &rarr; ~4.4-4.8% Oh43-absent rate) and explicitly left the residual "plausibly
genuine (sequencing error / true divergence / repeat confusion), not investigated
further." This investigates that residual directly, using a fresh regeneration of the
fixed (64-cap) refmap output plus independent evidence from minimap2 (aligned separately
to each of the 25 founder assemblies) and the pre-existing minimap2+liftover truth chain
in `ropebwt_refMap/validation/`.

**Scope**: the 1M-read Oh43 benchmark subset (`bench_ps4g_npy/reads/Oh43_1M.fastq`), the
same one all prior results in this directory use. Diagnose-only — no code changes.

## Reproducing the headline number

Re-ran the patched (64-cap) `ropebwt3` binary from scratch (`--ref-prefix=B73
--max-occ=-1 --lift ... --ps4g --npy --label-bed`, 36s, deterministic — status mix
EXACT 301,796 / PLACED 411,931 / MULTI 265,438 / UNPLACED 20,835, matching the prior
benchmark exactly). Computing the Oh43-absent rate directly from the fresh `.npy`
(256bp bins, any bin with &ge;1 credited founder-event but zero Oh43 count):

**33,793 / 668,108 bins with events are Oh43-absent = 5.06%** (bin-level; the 4.4-4.8%
figures in the prior doc used windowed/K=24 variants of the same underlying signal —
consistent, same ballpark).

Every one of these bins is defined by at least one **EXACT or PLACED** read that
positively excluded Oh43 from its credited gameteSet (0% of absent bins are "absent"
merely because only ambiguous MULTI/UNPLACED reads landed there) — **35,799 such
Oh43-excluding reads** in total (PLACED 25,974 / EXACT 9,825). The `n_carrier`
distribution for PLACED reads at these bins decays smoothly from 1 up to 25 with **no
pileup near the 64 cap** — confirms the carrier-cap fix is holding and this is not a
residual truncation artifact.

## Method

All 35,799 Oh43-excluding reads were pulled out and:
1. Aligned independently with `minimap2 -x sr` (mismatch/indel-tolerant, unlike
   refmap's FM-index exact/SMEM matching) separately to **Oh43.fa** and **B73.fa**
   (full 35,799-read set), and to all **25 founder FASTAs** (5,000-read random
   subsample, for tractability — 24s/genome, index-load-dominated).
2. Cross-checked against the pre-built, refmap-independent **minimap2-to-Oh43 +
   Oh43&rarr;B73-gVCF-liftover truth chain** (`validation/cache/truth_b73_Oh43.tsv.gz`),
   scored with the repo's own `validation/scripts/05_score_refmap.py`.
3. Compared per-read base quality (from the FASTQ) against a matched random sample of
   35,799 *non*-absent reads.
4. Checked spatial clustering (per-chromosome and per-1Mb-window absent-bin density,
   plus main-chromosome vs. unplaced-scaffold contigs).

## Result: it's not one thing — four distinct, quantifiable mechanisms

### 1. Genuine repeat/paralog mismapping — the dominant mechanism

The truth-chain cross-check is the clearest single number: **only 13.2%** of the 35,799
absent-bin reads have usable independent truth at all (vs. 57.7% for the whole read
population) — most are themselves repetitive/multi-mapping within the Oh43 assembly,
which is exactly the criterion the truth-chain excludes on. Of the 4,716 that *do* have
usable truth:

| | n | wrong-chromosome |
|---|---:|---:|
| Population baseline (`validation/RESULTS.md`) | 9.18M | 1.81% |
| **Oh43-absent-bin reads (this investigation)** | 4,716 | **69.7%** |

**A ~38x enrichment for genuine cross-chromosome mismapping.** This is corroborated
independently by the direct 25-founder minimap2 sweep, bypassing the truth chain
entirely: among absent reads with an *effectively unambiguous* best hit somewhere in the
25-genome panel (n_tied&le;2, i.e. at most one other founder ties, best identity
mean 98.2%), **only 11.5% pick Oh43** as that best hit — 88.5% clearly prefer a
different single founder, with no dominant "wrong" genome (best-founder counts spread
across ~20 different founders: B73 124, Oh43 75, Ki11 61, B97 58, CML247 56, ... —
diffuse, not a systematic confusion with one specific assembly). This matches the
diffuse founder-credit pattern already seen directly in refmap's own `.ps4g` output at
absent bins (top "other" founder credited only 5.5% of the time, next 5.2%, near-uniform
across all 24).

This diffuseness, plus **only 14.6%** chromosome agreement between minimap2's Oh43 hit
and refmap's B73 call even restricted to reads where Oh43 aligns as well as or better
than B73 (mean identity 99.1%), points at **shared repeat/transposon/IBD-block
sequence** rather than a specific systematic bug: the same short exact/near-exact
sequence occurs at many loci across the pangenome, refmap's FM-index-based placement
picks one (via `--lift` projection), and it frequently isn't the locus a gapped aligner
like minimap2 would call "primary." Spatial evidence supports this directly:

- **Unplaced scaffolds: 63.8% Oh43-absent** (3,543/5,552 bins) vs. **4.57%** on the 10
  main chromosomes (30,250/662,556) — scaffolds disproportionately hold repetitive,
  hard-to-assemble sequence, and contribute 10.5% of *all* absent bins from &lt;1% of
  total bins.
- Absent bins cluster sharply in specific 1Mb windows — chr5:200-202Mb, chr6:4-5Mb &
  21-23Mb, chr9:0-1Mb, chr8:160-161Mb — that **overlap the high-divergence windows
  already flagged independently** in `validation/RESULTS.md &sect;6` (chr5 200-202Mb,
  chr6 4Mb, chr9 1Mb all appear in both lists) — the same repeat hotspots, found twice,
  by two unrelated methods.

**Example** (`HWI-ST1348:...:14766:3923`): refmap calls it `EXACT` at
`B73_scaf_508:44267-44418`; independently, minimap2 places it uniquely (MAPQ=60) at
`chr6:160,172,507` — a collapsed/duplicated repeat that the B73 assembly represents
twice, once correctly on chr6 and once as an unplaced scaffold copy, with refmap's exact
match landing on the wrong copy.

**This mechanism dominates the residual** — it's the primary driver of both the
wrong-chromosome truth-chain result and the low same-chromosome agreement in the direct
minimap2 sweep, and it is a real, structural limitation of exact/near-exact FM-index
placement in repetitive genomic regions, not something `--max-occ` or the carrier cap
can fix (raising `--max-occ` only changes *whether* a repeated locus gets placed at all,
not which of several genuinely-tied loci is picked).

### 2. Sequencing error / low-quality reads

A large, separable chunk of absent reads simply don't align well *anywhere*:

| | mean per-read Phred | median |
|---|---:|---:|
| Absent-bin reads (n=35,799) | 20.5 | 24.5 |
| Matched non-absent reads (n=35,799) | 34.0 | 36.6 |

Cohen's d &asymp; 1.15 (large effect). Directly aligning to Oh43 and B73 with
minimap2, **24.9% of absent reads (8,909) fail to align well to either** ("poor
everywhere" — mean quality Q9.4, **median Q2.6**, essentially unusable reads), and in
the 25-founder sweep, **21.9%** match *none* of the 25 genomes at all. This is
independent of the repeat-confusion mechanism above: even among truth-chain-covered
reads, the wrong-chromosome subset skews to lower quality (mean Q22.0) than the
correctly-placed subset (Q28.0) — a bad base can break refmap's whole-read exact match
and knock a read into the more error-prone SMEM/carrier-projection fallback, consistent
with refmap's requirement for exact matches making it structurally more brittle to
sequencing error than a gapped aligner (this is the "PLACED, not EXACT" pathway; 72.6%
of absent-bin-defining reads are PLACED, vs. 41.2% in the whole population).

### 3. Genuine Oh43-vs-B73 divergence / possible Oh43 assembly-quality issues

A smaller (3.8%, 1,355 reads) but distinct bucket: B73 aligns clearly better than the
Oh43 assembly itself (mean identity gap ~5.3 points: Oh43 92.6% vs. B73 97.9%). Both
sides typically have modest MAPQ too, so this overlaps with mechanism 1 rather than
being fully separate, but it's a candidate signal for real Oh43-assembly gaps/errors at
specific loci (worth flagging to whoever maintains the Oh43 reference, not a refmap
issue).

### 4. A small, genuine reporting/assembly-coverage gap (structurally distinct)

**211 reads (0.6% of the 35,799)** are independently confirmed **correctly placed by
refmap** — right chromosome, exact position (`dist=0` in the truth-chain join), `EXACT`
status (i.e. the read matches B73 verbatim) — yet register **zero carriers at all**,
not just "Oh43 excluded, others present": `n_carrier=0` for literally all 211,
meaning *no* founder in the 25-assembly panel shares that exact 151bp sequence with B73
at that locus. Since the read demonstrably originates from Oh43 DNA and matches B73
perfectly there, this means the Oh43 (and every other founder's) *assembly*, as indexed
in the pangenome, doesn't carry the identical window at that exact position — most
plausibly a small assembly gap, scaffold break, or contig-boundary dropout in the
pangenome index at that specific locus. Small (~0.6% of absent reads &rarr; roughly
0.03pp of the ~5% headline rate) but a real, distinct, non-mismapping cause, echoing
(in miniature, and now confirmed rather than just hypothesized) the "assembly-gap/lift
dropout" mechanism the original root-cause investigation had ruled out for the *old*
8-cap bug specifically — it's real here, just small.

## Bottom line

| mechanism | evidence | rough share of absent reads |
|---|---|---:|
| Repeat/paralog mismapping (structural, not fixable via `--max-occ`) | 38x wrong-chr enrichment (69.7% vs 1.81%), 88.5% of unambiguous 25-way comparisons prefer another founder, scaffold absent-rate 63.8% vs 4.6%, recurring 1Mb hotspots matching independent validation | dominant (majority) |
| Sequencing error / low-quality reads | mean Q20.5 vs Q34.0 (d=1.15), 24.9% fail to align anywhere, wrong-chr reads skew lower-quality than correct ones | ~25% cleanly attributable, contributes further within mechanism 1 |
| Oh43-vs-B73 divergence / possible Oh43 assembly quality | B73 clearly better than Oh43 for 3.8% of reads | small, minor |
| Assembly-coverage/indexing gap (not mismapping) | 211 reads exactly correctly placed, literally zero carriers anywhere | ~0.6% of absent reads |

**Is 1-2% (matching the population's chromosome-level mismap rate) achievable for the
Oh43-absent metric specifically? No — and it shouldn't be, because the two numbers
measure different things.** The population-wide 1.81% wrong-chromosome rate is computed
over *all* reads, most of which land in unambiguous, uniquely-placeable regions. The
Oh43-absent metric is disproportionately populated (13.2% vs 57.7% truth coverage) by
reads that are hard to place *for anyone* — repetitive regions, scaffolds, low-quality
reads — precisely because a read that maps cleanly and uniquely to Oh43's own true locus
almost always credits Oh43 correctly (this is exactly what "PLACED via `--lift`" and
"EXACT" are designed to do, and do for the ~95% of Oh43 reads that aren't absent). The
absent bucket is, by construction, an enrichment for the genome's genuinely hard cases.

**What is fixable, in principle:**
- The repeat/paralog mismapping component is a real limitation of exact/near-exact
  FM-index placement + single-locus `--lift` projection in repetitive regions; a
  MAPQ-like confidence score reflecting *within-panel* multi-mapping (distinct from the
  existing k-mer-mode MAPQ, which is calibrated differently) would let downstream
  consumers (like the CRF) down-weight these calls rather than treating them as
  confident negative evidence for Oh43. This is the highest-leverage next step if
  further reduction is wanted, and squarely a `ropebwt3-phg` roadmap item, not a
  one-line fix.
- The 211-read assembly-coverage-gap class could be diagnosed further (which specific
  contigs/scaffolds are missing coverage at those loci) but is too small to matter for
  overall accuracy.
- The sequencing-error component is inherent to the 1x-coverage short-read data, not a
  tool limitation — deeper coverage or read-level quality filtering upstream would
  reduce it, not a refmap change.

## Reproducing

All intermediates for this analysis live in the session scratchpad (re-run refmap
output, minimap2 SAM files for Oh43/B73 full-set and 25-founder 5k-subsample, and the
classification/scoring scripts) — not promoted into the repo, per the diagnose-only
scope. The truth-chain scoring reused `validation/scripts/05_score_refmap.py` unmodified
against a qname-filtered copy of the fresh refmap TSV restricted to the 35,799
Oh43-absent-bin-defining reads.
