#!/usr/bin/env bash
# Pipeline: FASTQ -> ropebwt3 -> PHG convert -> GRITS impute -> bed-to-vcf
# Output VCF lands at OUT_DIR/vcf/out.vcf (matches what run_accuracy.sh expects)

set -uo pipefail

# maize chr ends : /workdir/irk9/data/phg-maize/target_vcf/chr_ends.txt
# cassava chr ends : /workdir/irk9/data/phg-cassava/target_vcf/chr_ends.txt

# ./run_grits2.sh --out-dir /workdir/irk9/data/phg-maize/test2/2x/grits --in-dir /workdir/irk9/data/phg-maize/test2/2x/grits/out/out.bed --chr-ends /workdir/irk9/data/phg-maize/target_vcf/chr_ends.txt --ref-vcf /workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz

# ./run_grits2.sh --out-dir /workdir/irk9/data/phg-cassava/target_vcf/1x/grits --in-dir /workdir/irk9/data/phg-cassava/target_vcf/1x/grits/out/VEN25_out.bed --chr-ends /workdir/irk9/data/phg-cassava/target_vcf/chr_ends.txt --ref-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_cleaned.vcf.gz

# ./run_grits2.sh --out-dir /workdir/irk9/data/phg-cassava/test2_redo/26.35x/grits --in-dir /workdir/irk9/data/phg-cassava/test2_redo/26.35x/grits/out/out.bed --chr-ends /workdir/irk9/data/phg-cassava/target_vcf/chr_ends.txt --ref-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_cleaned.vcf.gz

IN_FILE=""
OUT_DIR=""
REF_VCF=""
CHR_END=""
SAMPLE_BIN="/workdir/shared_files/sample/bin/sample"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir)         OUT_DIR="$2";         shift 2 ;;
        --ref-vcf)         REF_VCF="$2";         shift 2 ;;
        --in-dir)         IN_FILE="$2";         shift 2 ;;
        --chr-ends)         CHR_END="$2";         shift 2 ;;
        -h|--help)         exit 0 ;;
        *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
    esac
done


mkdir -p "${OUT_DIR}/out2"

python /workdir/irk9/data/phg-maize/bellas_scripts/fix_bed_boundaries.py "${CHR_END}" "${IN_FILE}" >  "${OUT_DIR}/out2/out.bed"

OUT_VCF="${OUT_DIR}/out2/out.vcf"

if [[ -f "${OUT_VCF}" ]]; then
    echo "[bed-to-vcf] Skipping — already exists: ${OUT_VCF}"
else
    echo "[bed-to-vcf] Converting to VCF..."
    module load java/21 2>/dev/null || true
    "${SAMPLE_BIN}" bed-to-vcf \
        --bed-dir "${OUT_DIR}/out2" \
        --reference-panel-vcf "${REF_VCF}" \
        --out-file "${OUT_VCF}" \
        || { echo "[ERROR] bed-to-vcf failed"; exit 1; }
    echo "[bed-to-vcf] Done: ${OUT_VCF}"
fi

echo ""
echo "All done. VCF: ${OUT_VCF}"


# /workdir/shared_files/sample/bin/sample bed-to-vcf --bed-dir /workdir/irk9/data/phg-maize/test2/0.1x/grits/out2 --reference-panel-vcf /workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz --out-file /workdir/irk9/data/phg-maize/test2/0.1x/grits/out2/out.vcf
