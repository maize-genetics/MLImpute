# --pav-agree sensitivity: what's the spread, and what does tightening it cost?

Follow-up to `pav_chain_read_fraction.md`. That measurement showed PAV anchoring
recovers 20.09% of reads that `refmap`/`chain`-without-`--lift` drops outright.
Given that, the question became whether `--pav-agree` (currently defaulted to
5000bp, the same value as `--pav-grid`) can be tightened toward the pipeline's
256bp bin size without losing much of that recovery.

## What `--pav-agree` actually gates

For a PAV-path read, `chain_emit_pav` projects every occurrence of the read's
longest assembly-only anchor onto the reference via the lift map, picks the
majority reference sequence, and takes the **median** position among the
projections that landed on it (`search.c:1218-1227`). `disp` is the **max absolute
deviation from that median** — a range-style spread, not a statistical variance
(no averaging or squaring). A read is emitted only if `disp <= --pav-agree`
(`search.c:1236`) *and* every projection agreed on the same reference sequence at
all (`bestn == np` — a separate, stricter gate that `--pav-agree` doesn't touch).

So: yes, `--pav-agree` is the knob that sets how much positional disagreement
among a read's own assembly copies is tolerated before the read is thrown away —
confirming the premise of the question — but it's a max-deviation cutoff, not a
variance threshold in the technical sense.

## Method

`disp` is computed for every read that clears the reference-agreement gate,
whether or not it currently passes `--pav-agree` — but the original code only
prints it for the survivors, so the rejected majority's spread was invisible.
Patched `chain_emit_pav` in the scratch build (not the user's checkout) to log
`disp`/`nmaj`/`np` unconditionally behind an opt-in env var
(`RB3_DBG_PAV_DISP=1`, one `fprintf(stderr,...)`, no other code path touched);
verified stdout from the instrumented binary is byte-identical to the
un-instrumented run (`diff -q` on both `.tsv` outputs). One rerun on the same 2M
Oh43 reads / v2 index+lift then gives the **full uncensored distribution** in a
single pass, rather than needing a separate `ropebwt3` run per threshold.

## The distribution has a heavy tail

589,934 candidates cleared the reference-agreement gate (`bestn==np`, `nmaj>0`) —
47% more than the 401,835 currently emitted at the default 5000bp cutoff. Of
those, 21.3% have `nmaj==1` (only one assembly occurrence projected — `disp` is
trivially 0, no real agreement evidence either way). Restricting to `nmaj>=2`
(464,346 reads, genuine multi-occurrence agreement):

| disp range | reads | % |
|---|---:|---:|
| 0 (exact agreement) | 119,371 | 25.7% |
| 1–100 bp | 75,316 | 16.2% |
| 101–1,000 bp | 27,751 | 6.0% |
| 1,001–5,000 bp | 53,809 | 11.6% |
| 5,001–50,000 bp | 153,935 | 33.2% |
| 50,001–1,000,000 bp | 21,619 | 4.7% |
| ≥1,000,001 bp | 12,545 | 2.7% |

It's not a smooth spread around a typical value — it's **bimodal**: a large tight
cluster near 0 (41.9% within 100bp) plus a heavy tail that reaches into the
hundreds-of-megabases (p99 = 92–119 Mb, i.e. individual "agreeing" occurrences
that actually disagree by nearly a chromosome's length — almost certainly repeats
or paralogous loci fooling the majority-reference-sequence vote, not real
positional noise around one locus).

## Survival curve: what tightening costs

| `--pav-agree` | reads emitted (of 401,835 @5000) | % of current PAV yield kept | % of all 2M input reads |
|---:|---:|---:|---:|
| 5000 (current) | 401,835 | 100.0% | 20.09% |
| 2500 | 375,077 | 93.3% | 18.75% |
| 1000 | 348,026 | 86.6% | 17.40% |
| **256** | **327,666** | **81.6%** | **16.38%** |
| 128 | 322,232 | 80.2% | 16.11% |
| 64 | 316,959 | 78.9% | 15.85% |
| 0 (exact) | 244,959 | 61.0% | 12.25% |

## Read

**Lowering `--pav-agree` to 256 — matching the pipeline's existing bin size —
keeps 81.6% of the reads PAV anchoring currently recovers**, for a ~20x tighter
positional claim (5000bp → 256bp). That's a real cost (74,169 reads, 18.4% of
current PAV yield) but not a large one relative to the 20x precision gain, and the
distribution explains why: a sizeable share of what gets cut at 256 was sitting in
the 5,001–50,000bp band (33.2% of nmaj≥2 candidates) or beyond — spread that's not
"one locus with a bit of jitter," it's disagreement large enough that snapping it
to a single 256bp bin would misrepresent it regardless.

Going further, to `--pav-agree=0` (exact-position match required), is markedly
more expensive: only 61.0% of current PAV yield survives — that's the point where
the trivially-agreeing `nmaj==1` reads (21.3% of all candidates, always pass
regardless of threshold) stop carrying the average, and the real cost of demanding
unanimous exact agreement shows up.

**Recommendation:** `--pav-agree=256` is a defensible match to `--bin-size=256`
that doesn't gut the recovery this feature exists for. `--pav-agree=0` would only
make sense alongside `--bin-size=1`, and even then throws away roughly 2 in 5 of
the currently-recovered reads.

## Caveats

- Single dataset (Oh43 inbred, 1x, v2 index), same as the parent measurement —
  not yet checked against another founder or read class.
- The heavy tail (>1Mb disp) suggests some fraction of "agreeing" reads are
  actually landing on repeats/paralogs that happen to share a reference
  sequence; not investigated further here — a candidate follow-up if the
  post-tightening false-positive rate ever needs auditing.
- This measures **count** surviving, not whether the *retained* reads are
  correctly placed — no ground truth was checked here (that's a separate,
  harder question requiring the simulated truth VCFs already in the corpus).

Raw counts: `results/pav_agree_disp_sensitivity.tsv`.
