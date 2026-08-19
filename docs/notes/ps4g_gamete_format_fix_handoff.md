# PS4G gamete-name-format fix — handoff (2026-08-19)

Status: **DONE, committed, pushed. Addressed one round of PR review feedback
(see "Review round" below) — waiting on further feedback.**

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
  `parse_gamete_header_line`/`parse_gamete_records` helpers. Detection was
  originally shape-based (3 tab fields, cols 2–3 integer) instead of
  requiring `:`; **superseded by the "Review round" section below**, which
  replaced shape sniffing with position-based (`#gamete`-section) detection.
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

## Review round (2026-08-19, later same day)

**PR feedback:** "ps4g.py and the rust script should only be parsing the
gamete info below the `#gamete` tag and not assume that the next thing has
the 3 column structure." I.e. the shape-sniffing from the original fix
above is exactly what needed to change — recognize gamete records by
*position* (inside the `#gamete\tgameteIndex\tcount` section), not by
column shape.

**What changed:**

- `crates/parser-core/src/ps4g.rs` — `parse_header_line` deleted, split
  into `is_gamete_section_tag` (recognizes the `#gamete` tag by its first
  tab field, case-insensitive), `parse_metadata_line` (the keyed lines:
  `#PS4G`, `#version=`, `#Command:`, `#TotalUniqueCounts:`), and
  `parse_gamete_record` (validates a line already known — by position — to
  be a record; no longer a shape sniff). Section state (`in_gamete_section`)
  now lives in `parse_ps4g`'s loop alongside `in_header`, which gained real
  meaning: `#` lines are only ever read while still inside the leading
  header block — a trailing `#`-comment after the data section starts is
  now an inert comment in both languages, where it previously could still
  be misparsed as metadata or a gamete record.
- `src/python/ps4g_io/ps4g.py` — new `_is_gamete_section_tag`,
  `_parse_metadata_key`, and unified single-pass `read_ps4g_header`
  (`parse_gamete_records` and `extract_metadata` both now delegate to it,
  replacing two separate full-file scans with one). A malformed line inside
  the gamete section is skipped with a `logging.warning`, not fatal, and
  doesn't close the section — so later well-formed records still parse.
- **No-tag fallback:** a file with gamete-shaped data but no `#gamete` tag
  no longer falls back to shape sniffing. Instead both parsers synthesize
  one gamete per distinct index found in the data section's `gameteSet`
  column, named by that index (Rust: post-loop pass over a per-index tally
  accumulated during the existing data-row loop; Python:
  `_synthesize_gametes_from_data`, a second linear scan only run when the
  header pass found nothing). This was an explicit user decision, not the
  "raise/error" default that would otherwise be the strict reading of the
  review comment.
- `docs/cli.md` — rewrote the Structure Overview and Metadata Lines sections
  to describe the `#gamete`-section boundary explicitly.
- Tests: Rust `mod tests` grew from 6 to 13 (six original `parse_header_line`
  unit tests adapted to the three new functions, seven new — tag-form
  recognition, malformed-record rejection, records-before-tag, stray
  3-column comment before/after the section, trailing comment inertness,
  no-tag synthesis, a full-file sanity check). Python `test_ps4g.py` grew
  from 26 to 33 (seven new, mirroring the Rust ones with `tmp_path`
  fixtures).

**Verification performed:**

- `cargo test -p parser-core` — 13/13 pass; `cargo build --workspace` clean.
- `pixi run -- pytest tests/python/ps4g_io/test_ps4g.py` — 33/33 pass.
- End-to-end against both real files in `data/` (outside the repo, at
  `/Users/zrm22/Desktop/gritsTests/data/`): `extract_metadata` /
  `build_index_lookup` still return 25 gametes, correct `version`/
  `total_reads`, and the same name ordering as before this round.
- `tests/python/bed_io/test_bed.py` — same pre-existing collection error as
  before (`ModuleNotFoundError: No module named 'src'`), unrelated,
  unchanged. `tests/python/ps4g_io/test_torch_loaders.py` has 3 pre-existing
  local-environment failures (OpenMP duplicate-runtime crash / DataLoader
  worker abort) confirmed present on the pre-review-round commit too —
  unrelated to this change.

## If resuming this thread

1. Check the GitHub PR for `fix/ps4g-gamete-name-formats` for further
   review comments.
2. If changes are requested: edit on this branch, re-run
   `pytest tests/python/ps4g_io/test_ps4g.py` and `cargo test -p
   parser-core`, commit, push.
3. The stacked branch `fix/ps4g-read-count-denominator` needs a rebase onto
   this branch's new commit (and a force-push) once this round is pushed —
   it was branched before this review round landed.
