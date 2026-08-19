# PS4G gamete-name-format fix — handoff (2026-08-19)

Status: **DONE, committed, pushed. PR open for review — waiting on feedback.**

## Problem

PS4G files from external tools (`ropebwt3 refmap`, PS4G v2.0) write gamete
header lines with a bare sample name (`#B73\t0\t784970`). Every GRITS parser
(Python and Rust) required a `:` in the name to recognize a gamete record
(`#B73:0\t0\t784970`), so colon-less files parsed to **zero gametes**:
`build_index_lookup` crashed with `ValueError: max() arg is an empty
sequence`, and the desktop/web viewer showed empty chromosome matrices.

Reported against `data/IDX-HYB__B73xCML103__0.1x/raw.ps4g` and
`data/IDX-HYB__Oh43xIl14H__0.1x/raw.ps4g` (25 gametes each, both colon-less).

## What changed

- `src/python/ps4g_io/ps4g.py` — new shared `SampleGamete` +
  `parse_gamete_header_line`/`parse_gamete_records` helpers. Detection is now
  shape-based (3 tab fields, cols 2–3 integer) instead of requiring `:`.
  `extract_metadata`/`build_index_lookup` take a `distinguish_gametes` flag
  (default `False`, preserves old bare-name behavior; `True` → `"B73:0"`
  style labels).
- Same copy-pasted colon-gated block removed from `hmm/hmm_impute.py`,
  `visualization/HMM_vis.py`, `bimamba/ps4g_eval.py`, `bimamba_impute_old.py`
  — all now call the shared helper.
- `crates/parser-core/src/ps4g.rs` — `parse_header_line` uses the same
  shape-based detection; accepts `#PS4G` (not just `##PS4G`). `GameteInfo`
  gained `sample_name`/`gamete_idx` (mirrored in `src/platform/types.ts`).
  `build_chromosome_matrix` only appends `:idx` to a display name when two
  gametes in the same panel actually collide on sample name.
- New fixture `data/sample_test_no_suffix.ps4g` (bare-name form); 12 new
  Python tests in `tests/python/ps4g_io/test_ps4g.py` (26 total); first-ever
  Rust unit tests for `parse_header_line` (6 tests) in `crates/parser-core/src/ps4g.rs`.
- `docs/cli.md` — documented both accepted gamete-name forms.

## Verification performed

- `pixi run -- pytest tests/python/ps4g_io/test_ps4g.py` — 26/26 pass.
- `cargo test -p parser-core` — 6/6 pass; `cargo build --workspace` clean.
- End-to-end against both real files in `data/`: `extract_metadata` /
  `build_index_lookup` / `convert_ps4g` now return 25 gametes and a proper
  `(N, 25)` matrix (was 0 / crash) for both `IDX-HYB__B73xCML103__0.1x` and
  `IDX-HYB__Oh43xIl14H__0.1x`.
- `tests/python/bed_io/test_bed.py` has a **pre-existing** collection error
  unrelated to this change (`ModuleNotFoundError: No module named 'src'` from
  `from src.python.ps4g_io...` in `bed_io/bed.py`) — confirmed present on
  `origin/develop` before this branch, not touched here.

## Branch / commit

- Branch: `fix/ps4g-gamete-name-formats`, based on `origin/develop`.
- Commit: `686709e` — "Accept both PS4G gamete-name forms (bare and
  ":idx"-suffixed)". Pushed to `origin`.
- `pixi.lock`'s local working-tree diff (pre-existing before this session,
  environment lockfile drift) was deliberately **not** included in the
  commit — it's unrelated to the PS4G fix.
- Plan file for this work: `~/.claude/plans/there-are-errors-in-polished-heron.md`
  (outside the repo, local to the session that did the work).

## If resuming this thread

1. Check the GitHub PR for `fix/ps4g-gamete-name-formats` for review
   comments.
2. If changes are requested: edit on this branch, re-run
   `pytest tests/python/ps4g_io/test_ps4g.py` and `cargo test -p
   parser-core`, commit, push.
3. No other open TODOs on this thread — everything in the original plan was
   completed and verified.
