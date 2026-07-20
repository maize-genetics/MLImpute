# Root cause: why ~11% of Oh43's own reads show up with Oh43-count=0

Follow-up to the 2026-07-14 real-data eval artifact, which found:
- 10.85% of test-split sites: Oh43 has no read at all → accuracy 0.641 there (vs 0.880 covered).

Open question at the time: *why would Oh43 reads, mapped against a pangenome that
includes the Oh43 assembly, ever fail to register an Oh43 count?*

## Answer: hardcoded 8-carrier cap in `ropebwt3-phg`'s `refmap_query()`, decoupled from `--max-occ`

File: `ropebwt_refMap/ropebwt3-phg/.claude/worktrees/refmap-ps4g-numpy/search.c`

```c
r->carriers = RB3_CALLOC(rb3_pos_t, 8);
for (i = 0; i < np && n_car < 8; ++i) { // keep a few distinct carrier sequences for reporting
    ...
    r->carriers[n_car++] = pos[i];
}
r->n_car_list = n_car, r->n_carrier = n_car;
...
if (p->gtab && n_car > 0) {   // PS4G/npy: carrier samples backing a (possible) PLACED call below
    r->gametes = RB3_MALLOC(int32_t, n_car);
    for (k = 0; k < n_car; ++k) r->gametes[k] = p->gtab->sid2g[r->carriers[k].sid>>1];
    r->n_gametes = n_car;
}
```

This is the **PLACED** path: a read that does *not* land an exact, unique match on the
B73 reference itself, but does match one or more non-reference "carrier" (founder)
sequences. `n_car` — the number of distinct founders recorded into the PS4G/npy gamete
set for that read — is capped at a compile-time constant of **8**, via a fixed 8-slot
array. Whichever 8 carriers happen to appear first in FM-index/suffix-array traversal
order are kept; any beyond that are silently dropped. This is a documented, intentional
simplification (`docs/usage.md:291`: "contributes the (up to 8) carrier samples"), but
it is **completely decoupled from `--max-occ`**, the flag this project explicitly set
to `-1` (→ auto = 25, the founder count) specifically so that loci shared by many
founders would still be placed rather than rejected as repeats (`search.c:583`,
`search.c:406`). Raising `--max-occ` controls *whether* a locus is placed at all; it
does nothing to the separate, independent 8-slot cap on *which* of the sharing founders
get reported once it is placed.

Net effect: whenever a read's true PLACED locus is shared by **more than 8** of the 25
founders, only 8 are recorded, chosen by an order with no relationship to genetic
identity — pure suffix-array traversal order. If Oh43 isn't among that arbitrary first
8, its own read is recorded as *not* supporting Oh43 at that site, even though it does.

By contrast, **EXACT**-status reads (matching the B73 reference itself) go through a
different branch that enumerates up to **64** equivalent hits into the same
`r->gametes`/`r->n_gametes` fields (`search.c` ref_found branch, `tmp[64]`) — high
enough that truncation is negligible for a 25-founder panel. So the failure mode is
specific to the *non-reference-matching* placement path.

## Evidence (converging, independent signals)

1. **Code**: the 8-cap is real, unconditional, and unrelated to `--max-occ` (confirmed
   by reading `search.c`, cross-checked against the `refmap-multi-speedup` worktree —
   identical cap — and the full git history of `search.c`, which shows no commit ever
   made this configurable; the one flag that sounded related, `--target-hits`
   (`fbe6a56`), is an unrelated early-stopping feature for bounding total reads
   processed, not carrier count).

2. **Structural spike in the matrix itself** (`data/real/ropebwt_oh43_1M.npy`, all
   1339 windows, 685,568 sites): among the 74,363 Oh43-absent sites (10.85% of the
   genome), the number of *other* founders present is **not** smoothly distributed —
   **56.06% land at exactly 8**, dwarfing every neighboring value (7 → 5.61%,
   9 → 0.51%). A real, unbounded biological signal (shared IBD blocks) would not
   produce a hard wall at one integer.

3. **No recurring founder cliques**: the "other=8" absent sites are spread across
   1334 of 1339 windows (essentially everywhere), and the specific 8-founder
   combinations recorded are close to unique — 38,264 distinct combos across 41,687
   rows, top combo occurring only 94 times. A true shared-haplotype/repeat signal
   would repeat the *same* founder clique; arbitrary SA-order truncation would not —
   and it doesn't.

4. **Direct raw-read confirmation**: the per-read refmap output for this exact
   1M-read Oh43 run (`ropebwt_refMap/bench_ps4g_npy/results/full_1M.ps4gonly.tsv`)
   shows **`PLACED, n_carrier=8` is the single largest PLACED bucket (143,313 reads)**
   — far ahead of every other carrier count (next is `n_carrier=1` at 55,312). Sampled
   rows confirm the mechanism directly, e.g.:
   ```
   HWI-ST1348:...:6124:2366  PLACED  8  P39,NC358,CML247,Tzi8,Ms71,Il14H,M162W,CML322 (all _chr5)  ...
   ```
   — an Oh43 read, placed on B73 chr5, reported as shared with 8 *other* founders,
   Oh43 itself absent from its own supporting record.

## What this is not

- **Not** sequencing error, assembly-gap/lift dropout, or repeat/paralog confusion in
  the biological sense (the mechanisms scoped in the original investigation plan) —
  those would show up as read-quality-dependent or genomically-clustered signals, not
  a hard spike at a single integer with near-uniform genome-wide spread and no
  recurring founder identity.
- **Not** a mislabeling/gamete-table bug: Oh43's own PS4G header total is the highest
  of all 25 founders (632,919 vs ~200-300k for others; `results_fixed/full_1M.ps4g`
  header), and the post-`contig_suffix()`-fix provenance of `data/real/ropebwt_oh43_1M.npy`
  is confirmed byte-for-byte against `results_fixed/full_1M.npy` (window 0 matches
  exactly under int8 clipping) — both candidate labeling/lift bugs from the original
  plan are ruled out.

## Consequence for the model

The 0.641-vs-0.880 accuracy gap is consistent with this mechanism, not contradictory:
at 8-capped PLACED sites the model still sees 8 *real*, genuinely-co-matching founders
(the truncation drops Oh43, not the others), so its "wrong" pick is very often still a
real IBD-consistent founder from the very same read — the same phenomenon the original
artifact already found dominating the *covered*-site errors (94.9% same-read/true-IBD).
This is an extension of that finding to the case where the truncation happens to
exclude the true label itself.

## Minor open item

The whole-genome Oh43-absent rate computed here (10.85%, all 1339 windows) matches the
artifact's reported figure exactly. Reproducing on just the deterministic 133-window
`eval.py` test slice gives 9.67% (same ballpark, expected window-to-window variance;
not investigated further since it doesn't affect the root-cause conclusion).

## Fix implemented and validated (2026-07-15)

Raised the hardcoded `8` to `RB3_RM_MAX_CARRIER = 64` (matching the existing `pos[64]`
locate buffer already used by `rb3_ssa_multi()`, so the cap can never truncate more
than the locate step itself does) at all 4 call sites in `search.c`
(`refmap_place_lift`'s `rsids`/`rposs`/`v` arrays and loop guard, and `refmap_query`'s
`car_seen`/`r->carriers` array and loop guard). Patch built and benchmarked in the
`refmap-ps4g-numpy` worktree; a from-scratch pre-edit build (via `git worktree add
HEAD --detach`) served as the "before" baseline for a fair, same-machine comparison.

**Speed** (1M-read Oh43 run, `--ref-prefix=B73 --max-occ=-1 --lift ... --ps4g --npy
--label-bed`, warm cache both runs): wall 35.9s → 36.2s (+0.9%), user CPU 635.7s →
639.1s (+0.5%), peak RSS 8.41 GB → 8.65 GB (+2.9%) — noise-level, confirming the
performance analysis (the cap gated a cheap step, not the FM-index locate).

**Correctness, raw per-read output**: the `PLACED, n_carrier=8` spike (143,313 reads,
previously the single largest PLACED bucket) is completely gone, replaced by a smooth,
naturally-decaying distribution (n_carrier=8→19,319; 9→18,532; 10→15,176; ...) — exactly
what a real, unbounded shared-locus distribution should look like, confirming the old
spike was pure truncation artifact, not biology. The concrete example read flagged
earlier (`...6124:2366`, chr5) now reports `PLACED, n_carrier=18`, **with Oh43 present**
in the gameteSet (previously 8 founders, Oh43 absent).

**Correctness, windowed matrix**: Oh43-absent rate (whole 1M-read genome, same windowing
script `crf/ropebwt_npy_to_matrix.py`) fell from **10.85% → 4.79%** — more than half of
the missing-Oh43 sites were the truncation bug; the residual 4.79% is plausibly genuine
(sequencing error / true divergence / repeat confusion, not investigated further). The
artificial "exactly 8 other founders" spike (56.06% of absent sites before) is gone.

**Model impact** (same sim-trained checkpoint, `checkpoints/haploid-sim/last.ckpt`, no
retraining, same real Oh43 test split, N=68,096 sites):

| | before (8-cap) | after (64-cap) |
|---|---:|---:|
| Overall Viterbi accuracy | 0.8567 | **0.9915** |
| Oh43-covered sites | 89.15% of sites, acc 0.880 | 95.60% of sites, acc **0.9971** |
| Oh43-absent sites | 10.85% of sites, acc 0.641 | 4.40% of sites, acc **0.9783** |

The gain is larger than "just" fixing the absent-site accuracy: covered-site accuracy
also jumped sharply (0.880→0.997), consistent with the cap previously truncating the
*competitor* founder set even at sites where Oh43 itself survived the cut — a smaller,
arbitrary competitor list gave the CRF a noisier picture everywhere the cap fired, not
only where it happened to exclude the true label.

**Artifacts from this validation**: patched binary and regenerated PS4G/npy/eval outputs
in the session scratchpad; validated copies of the fixed windowed matrices saved to
`grits_workdir/data/real/ropebwt_oh43_1M_fixed.npy` (K=25) and
`ropebwt_oh43_1M_fixed_k24.npy` (K=24, CML247 dropped, same as before) — originals left
untouched. New eval row `haploid-sim-on-ropebwt-oh43-k24-PATCHED` appended to
`results/eval.tsv`.

**Not yet done**: promoting the `_fixed` npy files to replace the originals, regenerating
against the full 16M-read Oh43 set (only the 1M-read benchmark subset was used here),
and committing/pushing the `search.c` change in the `ropebwt3-phg` worktree (currently an
uncommitted working-tree edit) — all left for explicit sign-off.
