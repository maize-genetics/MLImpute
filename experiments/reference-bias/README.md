# reference-bias

Does `refmap` under-call presence/absence variation (PAV) relative to a
reference-anchored `chain`+PAV path, and if so, by how much and why? Uses
`refbias_parse.py` (a genuine shared PS4G-parsing library — it lives in
`../simval-corpus/scripts/`, imported here via an explicit `sys.path`
entry, not duplicated).

## Key finding

`chain`'s raw B73 `hit_ratio` (65–75%) is mostly a PAV-path-mixing
artifact, confirmed at the source level. The corrected `hit_ratio_of_input`
is 34.7% (chain) vs. `refmap`'s 30.4% — a real gap, but only ~4–10
percentage points, not the ~30–40pp the raw numbers suggested. See
`results/refbias.md` (main writeup) and `results/refbias_pav_mechanism.md`.

## What's here

`refbias_eval.py`, `refbias_mash.py`, `refbias_report.py`. Results:
`results/refbias.md`, `results/refbias_pav_mechanism.md`,
`results/refbias_results.tsv`, `results/refbias_pav_fraction.tsv`,
`results/refbias_mash_residual.tsv`, `results/pav_agree_disp_sensitivity.md`,
`results/pav_chain_read_fraction.md` + their `.tsv` companions.
(`../simval-corpus/scripts/pav_accuracy_eval.py` and its
`pav_accuracy_eval_*` results are the general PAV-vs-refmap accuracy A/B
harness this investigation grew out of — that one stayed in
`simval-corpus` since it's a reusable 4-arm comparison tool, not specific
to the reference-bias mechanism question.)

## Running

Same environment as `../simval-corpus/README.md`.
