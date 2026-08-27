# PS4G Gametes-tab proportion fix — handoff (2026-08-19)

Status: **DONE, committed, pushed, rebased onto the base branch's PR-review
round (see "Rebase" below). Awaiting PR feedback.**

## Problem

The Gametes tab in the viewer divided each gamete's `read_count` by the sum
of all gametes' `read_count`. A PS4G data row's `gameteSet` can list several
gametes (a read maps to more than one founder), and the row's `count` is
credited to every gamete in that set — so summing per-gamete `read_count`
overcounts the true number of reads. Confirmed against
`data/sample_test.ps4g`: 4 actual reads (data-section `count` column
summed) vs. 7 summed per-gamete hits.

Separately, `GameteInfo.weight` was computed in `parse_header_line` against
the file's `#TotalUniqueCounts` header — a producer-declared value, not
always a true read count (the CRF window exporter,
`crf/export_windows.py`, writes the sum of all (site, founder) hits there
instead) — and was dead code the frontend never read.

## What changed

- `crates/parser-core/src/ps4g.rs` — `parse_ps4g` now sums the data
  section's `count` column into new `PS4GSummary.total_read_count` (the true
  read total, independent of header order or a disagreeing header value).
  `GameteInfo.weight` is computed in a pass after the whole file is parsed,
  as `read_count / total_read_count`, replacing the old
  `parse_header_line`-time computation against `#TotalUniqueCounts`.
- `crates/parser-core/src/types.rs`, `src/platform/types.ts` —
  `PS4GSummary` gained `total_read_count: u64`/`number`.
- `src/components/PS4GExplorer.tsx` / `.css` — Gametes tab now shows both
  metrics side by side: **% of Reads** (`read_count / total_read_count`,
  the corrected metric) and **% of Hits** (`read_count` / summed per-gamete
  `read_count`, the old math) — with stat cards and a caption explaining
  why they differ.
- `docs/cli.md` — documented both metrics and why `total_reads` is
  recomputed from the data section rather than trusted from
  `#TotalUniqueCounts`.
- Rust tests: 6 new (`total_read_count_sums_data_column_not_gamete_counts`,
  `weight_uses_computed_read_total`, `weight_is_independent_of_header_order`,
  `computed_total_ignores_disagreeing_header`, plus two adjusted for the new
  post-parse weight pass) — 12/12 pass in `parser-core`.

No Python-side change — `src/python/ps4g_io/ps4g.py` was out of scope for
this fix (the viewer's Gametes tab is Rust/TS only); worth checking whether
the Python loader has the same double-counting issue if it ever surfaces
per-gamete proportions.

## Verification performed

- `cargo test -p parser-core` — 12/12 pass (includes this fix's 6 new tests
  plus the gamete-name-format fix's 6 from the prior commit on this
  branch).
- `pixi run -- pytest tests/python/ps4g_io/test_ps4g.py` — 26/26 pass
  (unaffected by this change, re-run as a regression check).
- Reasoned through by hand against `data/sample_test.ps4g` (4 rows, 3
  gametes, one row with a 3-gamete `gameteSet`): total_read_count = 4,
  summed per-gamete read_count = 7, weights 1.0 / 0.5 / 0.25 for
  B73 / CML247 / W22 — matches the new Rust test fixtures.

## Branch / commit

- Branch: `fix/ps4g-read-count-denominator`, branched from
  `fix/ps4g-gamete-name-formats` (carries that branch's two commits:
  `686709e` fix, `832723f` handoff doc — see
  [[ps4g_gamete_format_fix_handoff.md]] / that branch's own PR).
- Commit: `a7d8e0a` — "Fix PS4G Gametes tab proportions double-counting
  reads". Pushed to `origin/fix/ps4g-read-count-denominator`.
- A stray `pixi.lock` working-tree diff (lockfile-format drift — local
  pixi 0.63.2 regenerated it as `version: 6` with the `platforms:` block
  stripped, vs. the committed `version: 7`) showed up unstaged at the
  start of this session; reverted with `git checkout -- pixi.lock`, not
  committed. Unrelated to this fix.

## Rebase (2026-08-19, later same day)

The base branch, `fix/ps4g-gamete-name-formats`, got PR review feedback
("recognize gamete records by position under the `#gamete` tag, not by
column shape") and landed a follow-up commit
(`59c70e4`, see [[ps4g_gamete_format_fix_handoff.md]]) that restructured
`crates/parser-core/src/ps4g.rs`'s header parsing — the same function
region this branch's `total_read_count`/weight change touches. This
branch was rebased onto that new base-branch tip and force-pushed:

- `crates/parser-core/src/ps4g.rs` — the post-loop weight pass and
  `total_read_count` accumulation from this fix were re-applied on top of
  the base branch's new `is_gamete_section_tag` /
  `parse_metadata_line` / `parse_gamete_record` split. No behavioral change
  to this fix's own logic — same weight formula, same
  `total_read_count` semantics — just carried forward onto the
  restructured file.
- `crates/parser-core/src/types.rs`, `src/components/PS4GExplorer.tsx`,
  `src/components/PS4GExplorer.css`, `src/platform/types.ts` — untouched by
  the base branch's change, carried forward unchanged.
- `docs/cli.md` — both branches' documentation edits combined (the
  `#gamete`-section explanation and the `% of Reads` / `% of Hits` table).
- Verified after rebase: `cargo test -p parser-core` — 18/18 pass (12 from
  this fix + 6 from the base branch's original set, now folded into the
  restructured functions); `cargo build --workspace` clean;
  `pixi run -- pytest tests/python/ps4g_io/test_ps4g.py` — 33/33 pass
  (unaffected, Python side untouched by this fix).

## If resuming this thread

1. Check GitHub for review comments on both PRs: the base
   `fix/ps4g-gamete-name-formats` and this branch,
   `fix/ps4g-read-count-denominator` (repo `maize-genetics/grits`). This
   branch is stacked on the other — if the base PR merges first, this one
   will need another rebase before it can merge cleanly.
2. If changes are requested: edit on this branch, re-run
   `cargo test -p parser-core` and
   `pixi run -- pytest tests/python/ps4g_io/test_ps4g.py`, commit, push.
3. No other open TODOs on this thread.
