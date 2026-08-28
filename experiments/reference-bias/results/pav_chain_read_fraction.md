# PAV chain read fraction (Oh43, 1x, v2 index)

Measures what share of reads take the `chain --lift` PAV-breakpoint path, before
deciding whether to wire `--ps4g`/`--npy` into it. Requested after reviewing
`pav-breakpoint-anchoring` (see plan: `the-next-task-i-humming-lobster.md`) turned
up that `chain`'s PAV emitters never call `rb3_ps4g_acc_add` — the npy/PS4G would
currently be empty regardless of read fraction, so this was purely "is the feature
worth building."

## Setup

- Branch: `origin/pav-breakpoint-anchoring` @ `0df7cad`, built fresh (not the
  user's dirty checkout).
- Reads: first 2,000,000 reads (of ~7.25M) from
  `Oh43.1.0x.R1.fastq.gz` in the simulated-validation corpus (v2 coordinates).
- Index/lift: v2 (`rope_bwt_index_v2/maizeFastaIndex_SampleContig_v2.{fmd,lift}`),
  matched to the v2-coordinate reads per `scripts/simval_paths.py`.
- Two runs, `-t 20`, `--ref-prefix=B73`:
  - **A**: `chain --lift=<v2.lift> --diverged-rows=pav` — every PAV-path read
    visibly tagged (default `--diverged-rows=ordinary` would hide the diverged
    half of the PAV path inside the ordinary-row pile).
  - **B**: `chain` with no `--lift` — baseline, reference-anchored rows only.
- The PAV fraction is a per-read property (independent of coverage), so 2M reads
  is enough; didn't need to scale to the full file.

## Results

| | reads | % of 2M input |
|---|---:|---:|
| ordinary (reference-anchored) row | 648,075 | 32.40% |
| **pav row, any class** | **401,835** | **20.09%** |
| — pav insertion (class 1) | 182,030 | 9.10% |
| — pav diverged (class 0) | 219,805 | 10.99% |
| any row at all | 1,049,910 | 52.50% |
| **no row at all** | **950,090** | **47.50%** |

Ordinary and pav row sets are disjoint (0 overlap) — expected, since
`chain_emit_pav` only runs when a read has zero reference SMEMs (`search.c:1315-1317`).
Confirmed independently: run B (no `--lift`) emits exactly 648,208 rows, matching
run A's ordinary-row count, and zero `pav:` rows.

**Reads recovered by PAV anchoring that baseline drops entirely: 401,835 (20.09%
of input).**

## Bin-concentration (input to the 256bp-vs-5kb question)

| | value |
|---|---:|
| distinct `(contig, 5kb-bin)` pav cells | 73,337 |
| mean rows/cell | 5.48 |
| median rows/cell | 2 |
| max rows/cell | 148 |
| cells with exactly 1 row | 25,165 (34.3% of cells) |
| rows sitting in cells with ≥5 rows | 309,737 (77.1% of pav rows) |

Mixed picture: most of the mass concentrates into a minority of cells (a handful
of true large structural variants, biggest cell 148 reads), but a third of cells
are singletons — reads that land near their own PAV region and nowhere else in
this sample. Consistent with plant-genome PAV having both hotspot indels and many
small/private ones.

## Sanity checks

- Build reports no warnings; `chain` usage text lists `--pav-grid`, `--pav-agree`,
  `--diverged-rows`.
- Baseline (no `--lift`) emits 0 `pav:` rows and its row count matches run A's
  ordinary count exactly.
- Since every read in this file is simulated purely from Oh43 (gamete index 21 in
  the fixed 25-founder alphabetical panel, confirmed against
  `bench_ps4g_npy/results/full_1M.new.npy.gametes.tsv`), every emitted row should
  carry gamete 21. 97.8% of ordinary rows and 98.8% of pav rows do; the remainder
  (14,457 ordinary / 4,991 pav) is consistent with simulated sequencing error
  breaking a SMEM segment onto a coincidentally-matching different assembly — not
  chased further, since it's background noise at the row level and this pass is
  about read-level yield, not per-row precision.

## Read

20% of WGS reads from a NAM founder have **zero exact match anywhere in the B73
reference** — not a small edge case. Given that, PAV anchoring recovers a fifth of
the read set that the current `refmap`/`chain`-without-lift path drops outright.
That argues the npy/PS4G wiring is worth doing.

The bin-concentration numbers say the flat 5kb grid is doing real, if partial,
work — most reads cluster into fewer, denser cells rather than each claiming its
own — but a third of cells are singletons, so collapsing everything to single-bp
resolution (as raised in conversation) would mostly just re-scatter that 77%
majority back into small pieces, since `--pav-agree` (the acceptance gate, default
5000) is far coarser than a 256bp/1bp bin would claim.

## Not measured here (deferred)

- Whether the 20% figure holds at other coverages/founders/hybrid or RIL classes
  — this run is Oh43-inbred-only, one coverage level.
- Whether tightening `--pav-agree` recovers usable precision at real cost to
  recall (would need a sweep, not covered by this quick pass).
- Full-file (~7.25M read) run — 2M was judged sufficient since the fraction is
  per-read, not coverage-dependent.

Raw counts: `results/pav_chain_read_fraction.tsv`.
