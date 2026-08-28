# cassava-diploid-crf

Synthetic-diploid CRF evaluation on the cassava pangenome (analogous to
`../tripsacum-diploid-crf/`, a different organism). Biggest affinity-model
gain observed of any organism tested (+43.5pp mean over the plain model);
the plain (non-affinity) model's accuracy was surprisingly low/variable —
unresolved. See `results/cassava_diploid.md` and
`results/cassava_diploid_affinity.md`.

## What's here

`cassava_diploid.py` + its results (`results/cassava_diploid*.{md,tsv}`).

## Running

Same environment as `../simval-corpus/README.md`. See the script's own
`--help` for its CLI; it follows the same real hap1×hap2-pair,
select-down-to-24-founders pattern as `tripsacum_diploid.py`.
