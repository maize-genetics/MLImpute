# crf-model-internals

Non-invasive probing of `GRITSCRFDiploid`'s internals (encoder outputs,
CRF transition/emission behavior, checkpoint comparisons) — hooks only,
the model source in `src/python/crf/` is never modified by anything here.
Covers three related but distinct investigations: encoder-output export
and visualization, a "tier 0/1/2" probe series into why the CRF loses
signal in low-support regions, and the whole-chromosome real-data decode
verification (`build_chr_mosaic_ril.py`, `whole_run_decode_check.py`).

## What's here

- **Export + plot**: `dump_model_internals.py`, `export_encoder_output.py`,
  `export_encoder_genomic_track.py`, `export_crf_inputs_genomic.py`,
  `plot_encoder_genomic_track.py`, `plot_crf_inputs_genomic.py`,
  `viz_grits.py`. **The 3 `plot_*`/`viz_grits.py` scripts need
  `matplotlib`, which is not in this repo's default `pixi` environment**
  — run them under `/home/zrm22/mambaforge/envs/phg-ml/bin/python`
  (the env the rest of this pipeline uses) or `pixi add matplotlib`.
- **Probing**: `probe_het_signal.py`, `homo_penalty_sweep.py`,
  `support_profile.py`, `compare_checkpoints.py`.
- **Whole-chromosome decode**: `build_chr_mosaic_ril.py`,
  `whole_run_decode_check.py` — verified: fixes small boundary artifacts,
  does not fix the deeper `homo_scale` mismatch or PAV-blip issues (see
  `results/wholechrom_decode_evaluation.md`).

## Results

Only findings docs + curated summary tables are committed here. The raw
exports these scripts produce (`model_internals/`, `encoder_export/`,
`crf_inputs_genomic*/`, `encoder_genomic_track*/` — mostly `.npz`
arrays and `.png` figures, hundreds of MB) are regeneratable via the
export/plot scripts above and were deliberately kept local-only — still
in `grits_workdir/results/` if you need the originals without re-running:
`results/tier0_findings.md`, `results/tier1_evaluation*.md`,
`results/tier2_evaluation*.md`, `results/t1024_*.md`,
`results/support_profile.{md,tsv}`, `results/wholechrom_decode_evaluation.md`.

## Running

Same environment as `../simval-corpus/README.md` (except the matplotlib
note above). These scripts generally take a checkpoint path and a
prepared `.npy` window as input — see individual docstrings for exact
flags; each was written for one specific probing question rather than a
uniform CLI.
