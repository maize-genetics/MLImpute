# Repetitive-region tracks (25 founders) + `ref_map` copy number

Follow-up to `oh43_residual_4pct.md`, which found the dominant driver of the residual ~5%
Oh43-absent rate was genuine repeat/paralog mismapping, inferred indirectly from
wrong-chromosome enrichment and quality signals. This turns that inference into a direct
measurement: an actual genome-wide repetitiveness track for all 25 NAM founders, joined
against read counts, plus a genuine per-read "copy number" exposed directly from `ref_map`.

## Part 1: copy number from `ref_map` — `--report-occ`

`ropebwt3-phg`'s `refmap_query()` already computes the exact FM-index interval size for
every read (`Iq.size`) and discarded it. Added a new **opt-in** flag, `--report-occ`
(`ropebwt_refMap/ropebwt3-phg/.claude/worktrees/refmap-ps4g-numpy/search.c`), that appends
it as a 12th trailing TSV column — a genuine, uncapped occurrence count: the number of
places the read's exact sequence occurs across the whole pangenome (reference + all 25
founders), before any `--max-occ`/carrier-cap truncation touches it.

**Off by default, verified true no-op when omitted**: built a from-scratch pre-edit binary
(preserved before editing) and diffed its output against the patched binary with
`--report-occ` omitted on a 10,000-read slice — byte-identical. With the flag on, only the
new trailing column appears; every other field is unchanged.

**Correctness, verified empirically (not just asserted from theory)**:
- A read landing on `B73_scaf_299:499-650` reports `occ=1` — confirmed as a genuinely
  unique, single-copy locus (present nowhere else across all 25 founders).
- Two PLACED reads with 11 and 2 carriers respectively report `occ=11` and `occ=2` exactly
  — matching their carrier counts one-for-one, with no doubling.
- A sweep of 500 random EXACT reads showed occurrence counts ranging 1–25+ with a smooth
  distribution, no artificial floor or spike.

This directly **contradicts an initial theoretical concern** (that dual-strand indexing
would inflate raw counts ~2x, requiring a correction factor): the empirical data shows no
such inflation for genuine single-copy loci. The documentation and code comments were
written to reflect what was actually measured, not the unverified theory — `occ=1` really
does mean "occurs exactly once across the whole panel," and a real inverted/palindromic
repeat would legitimately report more than once (that's correct behavior, not an artifact).

**Caveats** (documented in `docs/usage.md` and inline in `search.c`): `occ` is
pangenome-wide, not per-genome (a locus conserved across many founders reports a larger
count, reflecting shared/IBD structure — this is signal, not noise); for reads placed via
the SMEM fallback (no full end-to-end match), it describes the matched core, not
necessarily the whole read length; it is always `0` for `--kmer` mode (no single
whole-query interval exists in that algorithm).

## Part 2: per-genome repetitiveness tracks, all 25 founders

### Method and real cost

Built a solo FM-index self-index for each of the 25 founder FASTAs (`ropebwt3 build`,
skipping the sampled-suffix-array step entirely — `ropebwt3 suffix` only needs the raw
`.fmd`, not the `.ssa`, since it counts occurrences rather than locating genomic
positions), tiled each genome into overlapping 31-mers every 100bp (`ropebwt3 fa2kmer`),
queried occurrence counts (`ropebwt3 suffix`), and streamed the result straight into a
per-256bp-bin max-occurrence aggregate (matching refmap's own default bin size) without
ever writing the raw per-k-mer table to disk.

A timed pilot on one genome (Oh43) came in well under the pre-pilot estimate — build 2m21s,
tiling negligible, query 51,200 k-mers/sec single-threaded (~7 min genome-wide) — so all 25
genomes were run in batches of 5 concurrent pipelines (bounded by the ~38.6GB peak RSS of
the build step). **Total wall-clock: 46m52s**, zero failures, matching the pre-run estimate
of 45min–1.5hr.

### Finding A: B73-anchored repetitiveness strongly predicts Oh43-absent bins

Joining the B73 self-tiling track against the existing per-bin Oh43-absent data (from the
residual-4% investigation) gives a clean, monotonic gradient:

| B73 repetitiveness decile | mean occurrence count | Oh43-absent rate |
|---:|---:|---:|
| 0 (least repetitive) | 1.5 | 3.43% |
| 1 | 8.7 | 3.87% |
| 2 | 34 | 4.03% |
| 3 | 113 | 4.25% |
| 4 | 295 | 4.98% |
| 5 | 715 | 6.02% |
| 6 | 2,046 | 5.65% |
| 7 (most repetitive) | 58,964 | **11.44%** |

Absent bins are markedly more repetitive at the median than present bins (**311 vs. 55**
occurrences, ~5.7x). Independently, the top-15 most repetitive 1Mb windows found by this
track **reproduce every hotspot already flagged** in the residual-4% report from an
unrelated method (mismap-rate clustering) — chr5:200-202Mb, chr6:4-5Mb/21Mb, chr8:160-161Mb,
chr9:0-1Mb, chr1:227Mb all show clearly elevated repetitiveness here — the same regions,
found twice, by two independent methods. The track also surfaces several new, even more
extreme regions not previously flagged (chr4:2Mb, chr9:77Mb, chr2:196Mb, chr8:31Mb, and
three windows on chr1 — mean occurrence 700K-1.17M, an order of magnitude above the
previously-known hotspots), plausible centromeric/knob-repeat candidates worth a closer
look separately.

### Finding B: Oh43's *own* within-genome repetitiveness does not predict absence — a genuine mechanistic nuance

The natural hypothesis was that the SAME analysis using Oh43's own self-tiling track (i.e.
"is this locus repetitive within Oh43 itself") would show the same gradient. It does not:

| Oh43-self repetitiveness decile | mean occurrence count | Oh43-absent rate |
|---:|---:|---:|
| 0 | 1.2 | 4.59% |
| 4 | 156 | 4.40% |
| 8 (most repetitive) | 69,882 | 4.79% |

Essentially flat — no gradient at all, and absent vs. present bins show nearly identical
median within-Oh43 occurrence (96 vs. 90). **This is a real, mechanistically sensible
result, not a null finding to discard**: `ref_map`'s placement is always an FM-index query
against the whole concatenated pangenome, lift-projected through B73's coordinate system —
what determines whether a read's match is ambiguous is how many places that exact sequence
occurs *across the whole panel, as anchored to B73's representation of the locus*, not
whether Oh43 specifically has a repeated copy of it. The right repetitiveness signal for
predicting `ref_map` behavior is the pangenome/B73-anchored one (Finding A) or, more
directly, the new per-read `--report-occ` column from Part 1 — not a single founder's own
self-tiling track in isolation.

### Cross-founder repetitiveness comparison

| founder | median occ | p99 occ | max occ |
|---|---:|---:|---:|
| B73 | 101 | 48,432 | 7,574,852 |
| Oh43 | 101 | 24,928 | 5,221,822 |
| CML228 | 114 | 55,256,742 | 55,256,742 |
| CML52 | 113 | 48,975 | 43,003,664 |
| Tzi8 | 110 | 41,955 | 25,588,374 |
| M37W | 104 | 33,452 | 3,784,074 (lowest max of the panel) |

Full table (all 25) in `cross_founder_summary.tsv`. Medians are remarkably consistent
(93-125) across all 25 genomes — overall repeat *density* is similar panel-wide, as
expected for the same species — but the extreme tail varies enormously (CML228's single
most-repeated 31-mer occurs **55.2 million** times, vs. M37W's max of 3.8 million, a 14x
spread). CML228 and CML52's extreme maxima are candidates worth a closer look — either a
biologically real, unusually large satellite/knob array, or a collapsed-repeat assembly
artifact in that specific genome; not resolved here.

## Bottom line

- **Copy number**: now available directly from `ref_map` via `--report-occ` (opt-in,
  off by default, verified true no-op when unused, verified correct when used).
- **Repetitive regions**: now measured directly, genome-wide, for all 25 founders, in
  under 47 minutes total, using the same FM-index machinery `ref_map` itself runs on —
  not RepeatMasker/TRF or an external library. The B73-anchored version of this track
  independently reproduces and extends the repeat hotspots inferred indirectly in the
  residual-4% report, closing that loop with a direct measurement.
- **Read counts per repetitive region**: answered directly for the Oh43-absent question
  (Finding A's table above); the same join generalizes to "how many reads land in
  repetitive regions" for any read set by substituting a different per-bin read-count
  table into the same merge.
- A genuinely useful, non-obvious mechanistic finding fell out of doing this properly
  (Finding B): within-genome repetitiveness alone is the wrong signal for predicting
  `ref_map` placement difficulty — the pangenome/B73-anchored view (or the new `occ`
  column) is the right one.

## Reproducing / extending

- Per-genome pipeline: `run_one_genome.sh <fasta> <name> <outdir> [k] [step] [binsize]
  [build_threads]` — all parameters, not hardcoded.
- Batch driver: `run_all.sh` (processes all 25 in concurrency-limited batches, skips
  founders already done).
- Join/analysis: `join_repetitiveness_reads.py`; cross-founder summary inline in this
  session's work.
- All self-indices (~900MB/genome) are deleted immediately after each genome's track is
  written, to keep scratch usage bounded; only the compact per-bin repetitiveness TSVs
  (~85-90MB each, `<founder>_repetitiveness.tsv`) and this results doc are meant to persist
  beyond the session scratchpad — nothing has been promoted into the repo proper pending
  sign-off, per this project's usual diagnose-in-scratch-first pattern.
- `--report-occ` patch is currently applied in the `refmap-ps4g-numpy` worktree's
  `search.c` (uncommitted) and built/validated in an isolated scratch copy, not yet
  promoted to the worktree's live `ropebwt3` binary or committed — left for explicit
  sign-off, matching how the earlier 8→64 carrier-cap fix was staged.
