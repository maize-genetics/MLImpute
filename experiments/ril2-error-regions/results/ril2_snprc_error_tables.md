# IDX-RIL2 SNP+RefCall error rate — 3 filter-sweep tables

Same 5-pair × 5-coverage grid as the founder-path bp-weighted error tables
(see `windowing_hitfrac_binarize` memory / `test_crf_relatedness`
branch `windowing-quality-filters`), scored instead by **SNP+RefCall
error rate** (`compare_gvcf_truth_diploid.py --snp-refcall-metrics`):
sites where both truth and imputed calls classify to `{HOMREF, SNP}`
(indel-call sites on either side excluded). Reused all 75 already-decoded
`bed/` outputs — no re-alignment or re-inference, only `bed_to_vcf` +
comparator per row. Driver: `scripts/run_ril2_snp_scoring.py`. Raw
per-row results: `results/ril2_snprc_results.tsv`.

## UNFILTERED

| pair | 0.01x | 0.1x | 0.5x | 1.0x | 2.0x |
|---|---:|---:|---:|---:|---:|
| B73xOh43 | 0.0116% | 0.0091% | 0.0078% | 0.0104% | 0.0142% |
| B73xCML103 | 0.0128% | 0.0097% | 0.0159% | 0.0147% | 0.0148% |
| Oh43xIl14H | 0.0154% | 0.0103% | 0.0167% | 0.0286% | 0.0493% |
| B97xCML103 | 0.0219% | 0.0134% | 0.0215% | 0.0642% | 0.0731% |
| Il14HxB97 | 0.0051% | 0.0063% | 0.0105% | 0.0183% | 0.0566% |
| **mean** | **0.0134%** | **0.0098%** | **0.0145%** | **0.0272%** | **0.0416%** |

## MAX-HIT-FRAC 0.3

| pair | 0.01x | 0.1x | 0.5x | 1.0x | 2.0x |
|---|---:|---:|---:|---:|---:|
| B73xOh43 | 0.0192% | 0.0024% | 0.0040% | 0.0069% | 0.0132% |
| B73xCML103 | 0.0048% | 0.0073% | 0.0108% | 0.0144% | 0.0138% |
| Oh43xIl14H | 0.0155% | 0.0118% | 0.0167% | 0.0199% | 0.0286% |
| B97xCML103 | 0.0177% | 0.0083% | 0.0280% | 0.0308% | 0.0493% |
| Il14HxB97 | 0.0065% | 0.0039% | 0.0059% | 0.0149% | 0.0335% |
| **mean** | **0.0128%** | **0.0067%** | **0.0131%** | **0.0174%** | **0.0277%** |

## MAX-HIT-FRAC 0.5

| pair | 0.01x | 0.1x | 0.5x | 1.0x | 2.0x |
|---|---:|---:|---:|---:|---:|
| B73xOh43 | 0.0046% | 0.0057% | 0.0052% | 0.0037% | 0.0115% |
| B73xCML103 | 0.0189% | 0.0095% | 0.0119% | 0.0157% | 0.0174% |
| Oh43xIl14H | 0.0140% | 0.0146% | 0.0182% | 0.0301% | 0.0401% |
| B97xCML103 | 0.0164% | 0.0062% | 0.0517% | 0.0590% | 0.0706% |
| Il14HxB97 | 0.0011% | 0.0079% | 0.0138% | 0.0189% | 0.0383% |
| **mean** | **0.0110%** | **0.0088%** | **0.0202%** | **0.0255%** | **0.0356%** |

## Interpretation

- **Magnitude sanity check passed**: SNP+RefCall error (0.001–0.07%) runs
  10–100x lower than the founder-path bp-weighted error (0.1–4.9%) at
  every corresponding cell, as expected for a HOMREF-dominated genotype
  metric vs. a founder-switch metric.
- **The "error rises with coverage" pattern holds here too**, filter- and
  metric-independent — mean error roughly triples from 0.01x to 2.0x in
  all three tables, matching the founder-path tables' surprising trend.
  This rules out the founder-path metric itself as the source of that
  pattern.
- **B73xOh43 flips from the worst pair (founder-path tables) to one of
  the best (here)** — the opposite ranking. Consistent with the chr8
  root-cause finding (B73 and Oh43 share long IBD tracts on this
  chromosome): swapping between two founders whose actual sequence is
  nearly identical costs almost nothing in genotype/SNP terms, even
  though it counts as a 100% wrong call in the founder-identity metric.
  B97xCML103 is the worst pair by this metric instead.
