# Real-data diagnostic — true founder missing from reads at ~28% of sites

**Date:** 2026-06-24. **Reporter:** esb33 (CRF branch). **For:** zrm22 (data build).

## Summary
Both the cassava and maize pre-windowed diploid training matrices have a
**~72% "either-match" ceiling**: at only ~72% of sites does a read come from
**either** true founder (H1 or H2). Excluding ~5% sequencing error we expect
~95%+. The missing ~28% caps every downstream model — the diploid HMM scores
only **pair_acc 0.049 / hap_acc 0.215** on cassava, riding a per-site emission
ceiling of ~0.24.

This is **not** cassava-specific and **not** a label/index bug — see below.

## Files
- Cassava: `.../singleShuffledCassavaDataset/fullMaizeDataset_all_diploid.npy`
  (278,053 × 512 × 26; filename is mislabeled — content IS cassava). Outbred
  (H1≠H2 everywhere).
- Maize (control, "known-good"):
  `.../singleShuffledMaizeDataset/fullMaizeDataset_all_diploid.npy`
  (1,346,838 × 512 × 26). Inbred (H1==H2).
- Layout both: cols 0–23 = per-founder read counts (int8, 0–127),
  col 24 = H1, col 25 = H2. K=24 founders.

## Measurements (test-split sample, reads>0 sites)
| metric | cassava | maize |
|---|---|---|
| mean reads/site | 8.0 | 8.2 |
| frac sites with 0 reads | 0.000 | 0.000 |
| H1-founder has ≥1 read | 0.424 | 0.716 |
| H2-founder has ≥1 read | 0.416 | 0.716 (==H1, inbred) |
| **EITHER true founder has a read** | **0.721** | **0.716** |
| mean #founders-with-read per site | 4.97 | 4.42 |
| median rank of true H1 founder by read count (0=best of 24) | 7 | — |
| true H1 founder in top-3 by read count | 0.279 | — |
| per-site argmax ∈ {H1,H2} | 0.237 | — |

## Sim-vs-real true-founder presence (the cleanest framing)
"Either true founder (H1 or H2) carries a read at the site" — our simulator
hits the expected `1 − bad_frac` ≈ 96%; both real files fall ~24 points short.
(See `docs/notes/sfs_sharing.png`, right panel; script `crf/sfs_sharing.py`.)

| file | het | background (random founder) | H1 | H2 | **either H1/H2** |
|---|---|---|---|---|---|
| Sim inbred  | 0.00 | 20% | 96% | 96% | **96%** |
| Sim diploid | 0.96 | 33% | 64% | 64% | **96%** |
| Real maize  | 0.00 | 18% | 72% | 72% | **72%** |
| Real cassava| 1.00 | 21% | 42% | 41% | **72%** |

The sims are correct (true founder present ~96% = 1 − 5% error; the diploid sim's
64%/64%-individual but 96%-either is the intended interleaved single-gamete
design). Both real files sit at **72% either** — i.e. ~28% of read-bearing sites
have NO read from either labelled founder, far above the ~4% modelled error. The
background (random-founder) presence is similar across all four, so the deficit is
specific to the TRUE-founder channel: labels and reads disagree.

On the SFS *shape*: the real-data spectrum has a hard cutoff at K/2=12. **UPDATE
(2026-06-25): that cutoff is a FOLDING ARTIFACT, not the correct shape.** The
build folds the founder-match count to the minor side (support capped at K/2),
so a read matching K−2 founders is wrongly recorded as matching 2. The real
spectrum is therefore NOT a valid target either. The correct mini-haplotype
sharing spectrum is the *unfolded* neutral expectation: singleton-dominated with
n_i ∝ θ/i and a real tail extending to K (Ewens / coalescent SFS), never folded
at K/2. So BOTH sides were wrong: our sims are bell-shaped / over-shared (peak at
~K/A, no real singletons), and the real data is folded (artificial K/2 cap). The
simulator-realism fix is to derive the unfolded 1/i spectrum from theory (not to
match the folded real curve). This is a separate task from the true-founder
deficit below, which remains the real-data build problem.

## Ruled out: label/index misalignment
Shifting H1/H2 labels by ±1 makes either-match **worse** (0.72 → 0.36), so the
label columns are already optimally aligned to the founder feature columns. Not
an off-by-one, not a column swap.

## Interpretation
A read that truly originated from H1's chromosome should match H1's founder
almost always (that founder carries the allele by construction). A 28% miss
rate, present in BOTH datasets, points upstream to the matrix build, most likely:
1. **Read→founder match too strict / founder assemblies incomplete** — a
   mini-haplotype read spanning a SNP/SV absent from the true founder's assembly
   fails to match it (check `read_snps` / exact-match length, assembly coverage).
2. **Noisy answer-key** — the true founder path used as labels is partly wrong.

Either way it is upstream of the CRF/HMM model code: neither model can recover
founder signal that is not present in the read features.

## Asks for zrm22
- Confirm whether ~72% either-match is expected (imperfect nomination) or a bug.
- If a bug: check the read-matching length and founder-assembly completeness in
  the build (`build_training_data_diploid.py` / `extract_windows_single*.py`).
- A target either-match (after error) of ~0.95 would lift the per-site emission
  ceiling from ~0.24 toward usable, at which point CRF-vs-HMM on real diploid
  data becomes a meaningful test.
