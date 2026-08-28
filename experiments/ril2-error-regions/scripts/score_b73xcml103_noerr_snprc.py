#!/usr/bin/env python
"""
SNP+RefCall error-rate scoring (reusing run_ril2_snp_scoring.py's exact
prep_vcf + compare_gvcf_truth_diploid.py --snp-refcall-metrics call) for
IDX-RIL2 B73xCML103 binsize1-noerr @ 0.5x -- the dataset under
investigation for the chr5:86-134Mb chronic B73/CML103 oscillation zone.
Truth is fixed by (dataset_id, parents, replicate), independent of which
reads were used, so the official manifest's truth_h1/truth_h2 path is
reused directly without needing a (nonexistent) manifest row for the
"noerr" read variant.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "../../simval-corpus/scripts"))  # simval-corpus core modules (this file was moved from grits_workdir/scripts/)
import heldout_assembly_eval as hae  # noqa: E402
import simval_eval_one as seo  # noqa: E402
import simval_paths as P  # noqa: E402

PY = "/home/zrm22/mambaforge/envs/phg-ml/bin/python"
COMPARATOR = Path(__file__).parent / "compare_gvcf_truth_diploid.py"
TRUTH_GVCF = "/workdir/shared_files/grits_crf_evaluation/reads/maize/simulated_validation/truth/IDX-RIL2/B73xCML103.g.vcf.gz"

INDIVIDUAL = "B73xCML103"
COVERAGE = "0.5"
TAG = "binsize1-noerr"


def main():
    row_key = f"IDX-RIL2__{INDIVIDUAL}__{COVERAGE}x__{TAG}"
    outdir = P.SCRATCH_ROOT / row_key
    sample = f"{INDIVIDUAL}_ril2_{TAG}"
    out_dir = P.RESULTS_DIR / "ril2_snprc"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{row_key}.json"
    log_path = P.LOG_DIR / f"snprc_{row_key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    imputed_vcf = outdir / f"{sample}_imputed.vcf"
    filtered_vcf = outdir / f"{sample}_imputed.autosomes.vcf"
    imputed_vcf_gz = filtered_vcf.with_suffix(filtered_vcf.suffix + ".gz")
    if not imputed_vcf_gz.exists():
        if not filtered_vcf.exists():
            if not imputed_vcf.exists():
                hae.bed_to_vcf(outdir / "bed", str(P.PANEL_VCF_V2), imputed_vcf)
            seo.filter_to_autosomes(imputed_vcf, filtered_vcf)
            imputed_vcf.unlink(missing_ok=True)
        imputed_vcf_gz = hae.bgzip_and_index_vcf(filtered_vcf)
        filtered_vcf.unlink(missing_ok=True)
    t_vcf = time.time()
    print(f"VCF ready in {t_vcf-t0:.1f}s: {imputed_vcf_gz}")

    cmd = [PY, str(COMPARATOR),
           "--imputed-vcf", str(imputed_vcf_gz), "--sample", sample,
           "--truth-gvcf-h1", TRUTH_GVCF, "--truth-gvcf-h2", TRUTH_GVCF,
           "--partial-credit", "--class-breakdown", "--snp-refcall-metrics",
           "--json-out", str(json_path)]
    full_env = {**os.environ, "TMPDIR": str(P.TMPDIR)}
    print(f"running comparator, log -> {log_path}")
    with open(log_path, "w") as logf:
        logf.write(f"cmd={' '.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=full_env)
    dt = time.time() - t0
    print(f"comparator exit={proc.returncode} in {dt:.1f}s total")

    if proc.returncode != 0 or not json_path.exists():
        print(f"FAILED -- see {log_path}")
        return
    counts = json.loads(json_path.read_text())
    snprc_compared = counts.get("snprc_compared_sites", 0)
    snprc_error_rate = 1.0 - counts["snprc_gt_allele_matches"] / snprc_compared if snprc_compared else None
    snprc_partial_error_rate = 1.0 - counts["snprc_partial_credit_sum"] / snprc_compared if snprc_compared else None
    all_compared = counts.get("compared_sites", 0)
    all_error_rate = 1.0 - counts["gt_allele_matches"] / all_compared if all_compared else None

    print(f"\n=== {row_key} ===")
    print(f"SNP+RefCall: compared_sites={snprc_compared:,}  error_rate={snprc_error_rate:.4%}  "
          f"partial_error_rate={snprc_partial_error_rate:.4%}")
    print(f"  HOMREF: total={counts.get('snprc_class_total_HOMREF',0):,} mismatch={counts.get('snprc_class_mismatch_HOMREF',0):,}")
    print(f"  SNP:    total={counts.get('snprc_class_total_SNP',0):,} mismatch={counts.get('snprc_class_mismatch_SNP',0):,}")
    print(f"All-sites (for comparison): compared_sites={all_compared:,}  error_rate={all_error_rate:.4%}")


if __name__ == "__main__":
    main()
