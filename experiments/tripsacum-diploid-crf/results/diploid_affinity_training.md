# Diploid-affinity GRITS-CRF: first training run

## What was trained

`checkpoints/diploid-affinity-sim512-h3` — the first diploid-affinity checkpoint ever
trained (`train_diploid.py --founder-affinity`, previously untried). Same recipe as the
plain `diploid-sim512-h3` model (`--time-local-emis --lr 1e-4 --warmup-steps 500
--precision bf16-mixed --max-epochs 5 --homo-penalty 3`) plus `--spike-skip
--cosine-decay` as a bf16-stability guard, `--founder-affinity`, and
`--windows-per-individual 100`.

Training data was **regenerated**, not reused, because `--founder-affinity` needs windows
grouped into synthetic "individuals" with a restricted per-individual founder subset
(`simulate_alleles.py --windows-per-individual 100 --min-founders 2 --max-founders 24`,
otherwise identical to the plain model's data-gen recipe: `--sites 512 --min-crossovers 1
--max-crossovers 4 --inbreeding 0 --gamete-balance 0.5 --sharing-model coalescent
--windows 100000` → 1000 individuals × 100 windows, same total size as the plain model's
training set).

## Training: clean, no instability

All 5 epochs completed with no NaN/Inf/errors despite the flagged bf16-instability risk
(`docs/RESULTS.md`'s haploid E5 note). `val_pair_acc`: 0.576 (epoch 0) → 0.607 → 0.616 →
(epoch 3 not logged as best) → **0.618 (epoch 4, best)** — essentially matching the plain
model's own held-out number (0.6186) despite different training data.

## Evaluation: does affinity conditioning actually help?

Two same-data, apples-to-apples comparisons on the new dataset's held-out test split:

### `eval.py --mode diploid` (never feeds the model its own affinity signal — confirmed this
session `eval.py`'s diploid path never passes `ext_emb`)

| Model | pair_acc | hap_acc |
|---|---|---|
| diploid-affinity-sim512-h3 (its `ext_bias` unused here) | 0.5794 | 0.7322 |
| diploid-sim512-h3 (plain, never had this capability) | 0.5868 | 0.7363 |

Without its affinity signal, the affinity-trained model is marginally *worse* — expected:
its `ext_bias` module was trained to expect a real per-individual vector every time, so
`ext_emb=None` here is out-of-distribution for it, not a fair test of the mechanism.

### `eval_diploid_ties.py` (the mechanism-specific test — *does* feed the true `ext_emb`
via `make_diploid_affinity_splits`, and stratifies by within-window founder-tie count,
i.e. how many founders match the read at a site — higher tie count = harder, more
ambiguous emission)

| founders matching | plain hap_acc | affinity hap_acc | Δ |
|---|---|---|---|
| 1 (no tie) | 0.8909 | 0.8986 | +0.8 pp |
| 2-3 | 0.7549 | 0.7761 | +2.1 pp |
| 4-6 | 0.7328 | 0.7569 | +2.4 pp |
| 7-12 | 0.7317 | 0.7562 | +2.5 pp |
| 13+ | 0.7322 | 0.7564 | +2.4 pp |
| **all** | **0.7363** | **0.7601** | **+2.4 pp** |

## Verdict

**Affinity conditioning helped, modestly but consistently** — a genuine ~2-2.5 percentage
point hap_acc improvement across every founder-tie bin, when actually given its own
affinity signal (the only fair test of the mechanism). This is a *different* outcome than
the haploid E5 precedent flagged going in (`docs/RESULTS.md`: encoder-side affinity
conditioning didn't clearly beat baseline there, the win came from a decode-time cutoff
instead) — worth remembering both results rather than assuming one generalizes to the
other. The gain here is fairly uniform across tie bins rather than concentrated in the
high-tie bins as the evaluator's own docstring hypothesized ("if affinity is working, the
gain concentrates in the high-tie bins") — a real effect, just not shaped quite as expected.

## Not done here (flagged as follow-up, not silently skipped)

Pointing this checkpoint at the Tripsacum alignment tests
(`scripts/tripsacum_diploid.py`/`tripsacum_recombination.py`) — `run_diploid_eval` doesn't
currently compute/pass `ext_emb`; would need the same "single real-individual affinity"
pattern `eval.py` already implements for the haploid case (computing one genome-wide
`_founder_affinity` vector from the whole run's windows, since our Tripsacum runs aren't
pre-grouped into multiple `--windows-per-individual` blocks the way this training data is).
