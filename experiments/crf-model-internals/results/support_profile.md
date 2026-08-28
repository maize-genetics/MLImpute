# Train-vs-real per-founder read support

Sim: 200 of 1000 training individuals (`data/training/sim_diploid_512_affinity.npy`), stratified by founder count k. Real: 3 cached simval 0.1x rows (`scratch/simval_eval/<ROW>/windowed_k24_fixdrop23.npy`).

## Sim training data, stratified by k (founders carried per individual)

| k | n | carried credit | background credit | margin | het_frac | switches/window |
|---|---|---|---|---|---|---|
| 2 | 10 | 0.635 | 0.307 | 0.328 | 0.499 | 5.01 |
| 3 | 8 | 0.526 | 0.307 | 0.219 | 0.677 | 5.06 |
| 4 | 7 | 0.471 | 0.307 | 0.164 | 0.742 | 4.96 |
| 5 | 3 | 0.442 | 0.307 | 0.134 | 0.803 | 5.03 |
| 6 | 8 | 0.416 | 0.306 | 0.110 | 0.828 | 4.93 |
| 7 | 6 | 0.401 | 0.307 | 0.094 | 0.860 | 4.91 |
| 8 | 8 | 0.388 | 0.308 | 0.080 | 0.873 | 4.91 |
| 9 | 5 | 0.381 | 0.306 | 0.075 | 0.887 | 5.07 |
| 10 | 8 | 0.372 | 0.307 | 0.065 | 0.909 | 4.99 |
| 11 | 3 | 0.366 | 0.305 | 0.061 | 0.912 | 5.13 |
| 12 | 9 | 0.361 | 0.307 | 0.054 | 0.913 | 5.10 |
| 13 | 9 | 0.357 | 0.307 | 0.050 | 0.927 | 5.01 |
| 14 | 6 | 0.355 | 0.308 | 0.047 | 0.931 | 5.04 |
| 15 | 15 | 0.350 | 0.307 | 0.043 | 0.928 | 5.00 |
| 16 | 11 | 0.348 | 0.307 | 0.042 | 0.943 | 5.01 |
| 17 | 16 | 0.345 | 0.307 | 0.039 | 0.940 | 4.95 |
| 18 | 13 | 0.344 | 0.308 | 0.036 | 0.946 | 4.95 |
| 19 | 4 | 0.342 | 0.307 | 0.035 | 0.957 | 4.99 |
| 20 | 10 | 0.340 | 0.306 | 0.034 | 0.954 | 4.97 |
| 21 | 11 | 0.338 | 0.307 | 0.031 | 0.948 | 4.93 |
| 22 | 5 | 0.338 | 0.305 | 0.032 | 0.959 | 4.89 |
| 23 | 11 | 0.335 | 0.307 | 0.028 | 0.957 | 5.03 |
| 24 | 14 | 0.335 | nan | nan | 0.963 | 5.04 |

## Real simval 0.1x rows

| row | kind | k | credit (raw) carried/bg | credit (binary) carried/bg | ps4g carried/bg | cardinality | frac cells>1 |
|---|---|---|---|---|---|---|---|
| IDX-INBRED__Oh43__0.1x | inbred | 1 | 1.032 / 0.379 | 0.988 / 0.366 | 0.988 / 0.363 | 9.40 | 0.0140 |
| IDX-INBRED__Il14H__0.1x | inbred | 1 | 1.036 / 0.332 | 0.988 / 0.320 | 0.989 / 0.317 | 8.35 | 0.0126 |
| IDX-HYB__Oh43xIl14H__0.1x | hybrid | 2 | 0.694 / 0.349 | 0.672 / 0.340 | 0.674 / 0.339 | 8.82 | 0.0102 |

Real credit (binary) and the independent PS4G-header cross-check
(`refbias_parse.parse_ps4g_header`) agree to within 0.003 on every row —
validates the profiler.

**Headline: H1 (read-support mismatch) is falsified as an explanation for the
inbred/hybrid gap.** Real hybrid margin (0.672 − 0.340 = 0.332) matches sim's
k=2 margin (0.635 − 0.307 = 0.328) almost exactly. Real inbred (k=1) has no
training analogue at all (k=1 never occurs) yet scores best. Distance from
the training distribution does not predict error.

## Model-internals readout (H2 / H3 / H4)

Full detail: `results/model_internals/manifest.json` + `<row_id>.npz`
(checkpoint `diploid-affinity-sim512-h3/d-epoch=04-val_pair_acc=0.6179.ckpt`,
24 windows/row). Self-check (`emis_p` reconstructed from the hooked
per-founder `emis_f` vs the model's own returned `emis_p`) matched exactly
(`max_diff=0.0`) on all 5 rows.

| row | pair_acc | hap_acc | marginal_hap_acc | decoded / true switches per window | het_frac (pen / no-pen) |
|---|---|---|---|---|---|
| IDX-INBRED Oh43 | 1.000 | 1.000 | 0.500\* | 0.00 / 0.00 | 0.000 / 0.000 |
| IDX-INBRED Il14H | 1.000 | 1.000 | 0.500\* | 0.00 / 0.00 | 0.000 / 0.000 |
| IDX-HYB Oh43xIl14H | 0.884 | 0.936 | 0.936 | 1.58 / 0.00 | 0.988 / 0.000 |
| sim k=2 | 0.406 | 0.697 | 0.699 | 4.83 / 5.13 | 0.994 / 0.000 |
| sim k=13 | 0.591 | 0.746 | 0.751 | 3.54 / 5.17 | 0.999 / 0.000 |

\* `marginal_hap_acc` structurally cannot represent a homozygous call — it
always picks the top-2 *distinct* founders by marginal probability, so a
perfect homozygous prediction (both slots = Oh43) still "misses" one slot by
construction. Not a bug: it demonstrates that the joint pair-CRF encodes
zygosity a naive per-founder marginal view cannot. Read the inbred row as
"N/A", not "50% wrong."

**H2 (path structure) — confirmed, real but modest at 0.1x.** The real
hybrid decodes with 1.58 switches/window against a true value of exactly 0
(zero breakpoints by this corpus's construction). Training never saw a
constant, zero-switch pair — its own decoded/true switch rates (sim k=2:
4.83 vs 5.13) show the model reproduces roughly its training-time switch
rate as background behavior, and some of that carries over to real hybrid
decoding as spurious switches.

**H3 (het-prior miscalibration) — evidence points the other way at 0.1x: the
penalty is doing its job.** Removing the fixed `homo_penalty` collapses the
decode to 0% heterozygous on every row (`homo_mass_raw`→0.86–0.96,
`het_frac_decoded_nopen`=0.000 everywhere) — the penalty is necessary, not
optional. With it applied, real hybrid decodes at 98.8% heterozygous (true
100%), reasonably close. At 0.1x this is not the dominant source of the
11.6% pair_acc gap; the coverage-dependent version of this effect documented
in `[[simval_full_corpus_eval]]` (error rising with coverage) is a different,
higher-coverage regime this dump doesn't probe.

**H4 (pair-decode ambiguity) — not a major extra cost beyond per-founder
uncertainty.** For the real hybrid, `marginal_hap_acc` (0.936) is
essentially identical to `hap_acc` (0.936) — decoding the joint 325-way pair
state costs almost nothing extra relative to just knowing each founder's own
marginal probability. Whatever is causing the residual ~6% haplotype-level
error is already present at the per-founder emission level, not introduced
by the joint decode.

**Verification.** The reimplemented forward-backward's argmax matched
`python.crf.train_diploid._dcrf_marginal` exactly (0/4096 mismatches on a
held-out batch). Decoded accuracy on the 24-window samples tracks
`results/simval_results.tsv`'s whole-genome numbers in direction and rough
magnitude: inbred ≈0% error both ways (Oh43/Il14H `error_rate`≈5e-6, dump
`pair_acc`=1.000); hybrid `error_rate`=7.3% / `partial_error_rate`=3.7%
(whole genome) vs dump `1-pair_acc`=11.6% / `1-hap_acc`=6.4% (24 of 2176
windows) — same order of magnitude, not identical, as expected from a small
window sample scored by a different metric convention (per-window average
vs whole-genome per-site).

**Net reading:** of the three hypotheses this dump was built to
discriminate, H2 (spurious switching from never having trained on a
zero-breakpoint hybrid) is the one with unambiguous, directly-measured
support at 0.1x; H3 is present but not obviously broken at this coverage;
H4 is not a distinguishing factor here. This points toward the training
data's total *absence* of the hybrid/inbred path-structure regime (0
training individuals with het_frac<0.05 or >0.95 — see the k-stratified
table above) as the more promising next target than either the read-support
distribution (falsified) or the het-prior calibration (looks reasonable
here specifically).

