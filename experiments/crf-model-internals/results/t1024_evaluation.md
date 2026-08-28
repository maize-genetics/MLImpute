# T=1024 window-size experiment: evaluation and verdict

## What was tested

Combined experiment: Tier 1 (sparse crossovers, `--min-crossovers 0
--max-crossovers 1`) + Tier 2 (`--constant-pair-frac 0.3
--constant-inbred-frac 0.5`) simulator settings, but at `--sites 1024`
instead of 512, with `--ancestor-crossovers 16` (scaled 2x from the T=512
default of 8, per `docs/RESULTS.md`'s E12-fix precedent). Recipe otherwise
identical to Tier 1/2: `--founder-affinity --learned-het --time-local-emis`,
lr 2e-5, grad-clip 0.3, cosine-decay, spike-skip, warmup 500, bf16-mixed,
3 epochs. Run name `breedpop-sparse-constant-t1024-affinity-learnedhet`.

This was scoped as the one genuinely new variable from E11-affinity
(`docs/RESULTS.md`'s best documented recipe used T=1024) that we hadn't
tried, without reintroducing E11-affinity's dense-crossover regime (already
diagnosed and fixed by Tier 1 as the H2 spurious-switching cause).

## Training did not converge

```
Epoch 0: val/pair_acc=0.276  val/loss=23.4
Epoch 1: val/pair_acc=0.278  val/loss=23.4  (best checkpoint, by monitor)
Epoch 2: val/pair_acc=0.278  val/loss=47.0
```

`val_pair_acc` is flat at ~0.278 for the entire run (vs Tier 1's 0.23→0.79
and Tier 2's climb to 0.92 by epoch 1, same recipe at T=512), and
`val_loss` *increases* monotonically rather than decreasing. Per-site,
final val_loss/T is ~2x Tier 2's per-site rate (47.0/1024=0.046 vs
11.7/512=0.023). The same lr/grad-clip/warmup that converges well at T=512
did not transfer to T=1024 (4x self-attention FLOPs/layer, and the CRF's
per-window NLL roughly doubles in scale) — this looks like a genuine
training-hyperparameter mismatch, not evidence about window size itself.

## Real-data comparison (cheap `dump_model_internals.py`, n=50 windows/row)

Full 4-way table with `diploid-affinity-sim512-h3` (baseline),
`breedpop-sparse-affinity-learnedhet` (Tier 1), and
`breedpop-sparse-constant-affinity-learnedhet` (Tier 2) in
`results/t1024_evaluation_raw.md`. Headline numbers:

| row | metric | baseline | Tier1 | Tier2 | **T1024** |
|---|---|---|---|---|---|
| Oh43 (inbred) | pair_acc | 1.000 | 0.840 | 0.840 | **0.636** |
| Il14H (inbred) | pair_acc | 1.000 | 0.900 | 0.900 | **0.805** |
| Oh43xIl14H (hybrid) | pair_acc | 0.828 | 0.618 | 0.605 | **0.852** |
| Oh43xIl14H (hybrid) | homo_mass_pen | 0.003 | 0.264 | 0.261 | **0.033** |
| RIL | pair_acc | 0.546 | 0.683 | 0.697 | **0.748** |
| RIL homo/het gap | -0.575 | +0.083 | +0.058 | **-0.288** |

The hybrid/RIL pair_acc numbers look like wins at first glance, but
`homo_mass_pen` and the RIL homo/het calibration gap show T1024 is **not**
better-calibrated than Tier 1/2 — it sits at an intermediate point between
baseline's extreme heterozygous-bias (never predicts homozygous) and
Tier 1/2's genuine calibration fix, still biased toward "always
heterozygous" (gap -0.288, vs Tier1/2's near-zero +0.06/+0.08). That bias
happens to score well on rows that are truly heterozygous (hybrid, most of
RIL) and costs badly on rows that are truly homozygous (both inbred rows,
worse than Tier 1/2 and far worse than baseline). Combined with the
non-converged training curve, this reads as the model getting stuck in a
partial, imbalanced state early in training — not a genuine effect of the
larger window giving the encoder more disambiguating context.

## Verdict: inconclusive, not promoted

Per the gating decision agreed on before this run (full real-pipeline
check only proceeds if the cheap signal shows a genuine improvement), the
full pipeline check was **not run**. The apparent hybrid/RIL improvement is
confounded by a training run that did not converge, so this experiment
does not answer whether T=1024 (bigger encoder context) helps the
hybrid/heterozygous accuracy problem — it needs a retrain with
hyperparameters actually tuned for T=1024 (lower lr and/or longer warmup,
possibly reduced batch size given ~4x attention cost) before the question
can be answered cleanly. Not pursued further this session; flagging as the
natural next step if window-size is revisited.

`marginal_hap_acc=0.5000` appearing for both inbred rows across *all four*
checkpoints (not just T1024) is not a bug — it's structural: the top-2
marginal-founder metric always returns two distinct founder indices, so the
best achievable score against a homozygous truth (same founder twice) is
exactly 0.5.
