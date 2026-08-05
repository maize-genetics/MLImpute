#!/usr/bin/env bash
# Pipeline: FASTQ -> ropebwt3 -> PHG convert -> GRITS impute -> bed-to-vcf
# Output VCF lands at OUT_DIR/vcf/out.vcf (matches what run_accuracy.sh expects)

set -uo pipefail

# ============================================================
# ARGUMENT PARSING
# ============================================================

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Required:
  --fastq          FILE  Input FASTQ file (.fastq.gz)
  --index-fmd      FILE  ropebwt3 index .fmd file
  --phg-bin        FILE  PHG binary (e.g. /path/to/phg/bin/phg)
  --splines-dir    DIR   Spline knot directory
  --out-dir        DIR   Output directory (VCF -> OUT_DIR/out/out.vcf)
  --ref-vcf        FILE  Reference panel VCF for bed-to-vcf
  --model-ckpt     FILE  GRITS model checkpoint (.pt file)
  --chr-ends       FILE  Chromosome boundary file for fix_bed_boundaries.py

Optional:
  --sample-name    STR   Sample name [default: FASTQ basename without .fastq.gz]
  --threads        INT   Threads for ropebwt3 [default: 4]
  --min-mem-length INT   Min MEM length [default: 148]
  --max-num-hits   INT   Max number of hits [default: 12]
  --ropebwt3       FILE  ropebwt3 binary [default: /workdir/irk9/software/ropebwt3/ropebwt3]
  --sample-bin     FILE  sample binary [default: /workdir/shared_files/sample/bin/sample]
  --grits-py       FILE  GRITS impute.py [default: /workdir/irk9/software/grits/src/python/impute.py]
  --conda-env      STR   Conda environment for GRITS [default: grits2]
  --conda-base     DIR   Miniconda base directory [default: /workdir/irk9/software/miniconda3]
  -h, --help             Show this help and exit
EOF
}

FASTQ=""
INDEX_FMD=""
PHG_BIN=""
SPLINES_DIR=""
OUT_DIR=""
REF_VCF=""
MODEL_CKPT=""
CHR_END=""
SAMPLE_NAME=""
THREADS=4
MIN_MEM_LENGTH=148
MAX_NUM_HITS=12
ROPEBWT3="/workdir/irk9/software/ropebwt3/ropebwt3"
SAMPLE_BIN="/workdir/shared_files/sample/bin/sample"
GRITS_PY="/workdir/irk9/software/grits/src/python/impute.py"
CONDA_ENV="grits2"
CONDA_BASE="/workdir/irk9/software/miniconda3"

# ./grits.sh --fastq /workdir/irk9/data/phg-maize/bellas_scripts/testing_suite_bella/testing_suite_test/reads/B97.fastq.gz --index-fmd /workdir/smm477/phg-maize/ropebwt-index-ML/ropebwt_index.fmd --phg-bin /workdir/irk9/software/phg_v2/build/distributions/phg/bin/phg --out-dir /workdir/irk9/data/phg-maize/bellas_scripts/testing_suite_bella/testing_suite_test/grits --ref-vcf /workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz --model-ckpt /local/workdir/zrm22/HackathonJun2026/NonCollapsedDatasets/seq2seqOutput/best_model.pt --chr-ends /workdir/irk9/data/phg-maize/target_vcf/chr_ends.txt

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fastq)           FASTQ="$2";           shift 2 ;;
        --index-fmd)       INDEX_FMD="$2";       shift 2 ;;
        --phg-bin)         PHG_BIN="$2";         shift 2 ;;
        --splines-dir)     SPLINES_DIR="$2";     shift 2 ;;
        --out-dir)         OUT_DIR="$2";         shift 2 ;;
        --ref-vcf)         REF_VCF="$2";         shift 2 ;;
        --model-ckpt)      MODEL_CKPT="$2";      shift 2 ;;
        --chr-ends)        CHR_END="$2";         shift 2 ;;
        --sample-name)     SAMPLE_NAME="$2";     shift 2 ;;
        --threads)         THREADS="$2";         shift 2 ;;
        --min-mem-length)  MIN_MEM_LENGTH="$2";  shift 2 ;;
        --max-num-hits)    MAX_NUM_HITS="$2";    shift 2 ;;
        --ropebwt3)        ROPEBWT3="$2";        shift 2 ;;
        --sample-bin)      SAMPLE_BIN="$2";      shift 2 ;;
        --grits-py)        GRITS_PY="$2";        shift 2 ;;
        --conda-env)       CONDA_ENV="$2";       shift 2 ;;
        --conda-base)      CONDA_BASE="$2";      shift 2 ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "ERROR: Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

# Validate required args
for var_name in FASTQ INDEX_FMD PHG_BIN SPLINES_DIR OUT_DIR REF_VCF MODEL_CKPT CHR_END; do
    if [[ -z "${!var_name}" ]]; then
        echo "ERROR: --$(echo "${var_name}" | tr '[:upper:]_' '[:lower:]-') is required" >&2
        usage; exit 1
    fi
done

# Derive sample name from FASTQ basename if not provided
if [[ -z "${SAMPLE_NAME}" ]]; then
    SAMPLE_NAME=$(basename "${FASTQ}" .fastq.gz)
    SAMPLE_NAME="${SAMPLE_NAME%.fastq}"
fi

mkdir -p "${OUT_DIR}" "${OUT_DIR}/out" "${OUT_DIR}/logs"

# -----------------------------
# Log file — captures all stdout/stderr for this run
# -----------------------------
LOG_FILE="${OUT_DIR}/logs/${SAMPLE_NAME}_grits.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "[grits pipeline] Started:     $(date)"
echo "[grits pipeline] Sample:      ${SAMPLE_NAME}"
echo "[grits pipeline] FASTQ:       ${FASTQ}"
echo "[grits pipeline] Index:       ${INDEX_FMD}"
echo "[grits pipeline] Out dir:     ${OUT_DIR}"
echo "[grits pipeline] Ref VCF:     ${REF_VCF}"
echo "[grits pipeline] Model:       ${MODEL_CKPT}"
echo "[grits pipeline] Chr ends:    ${CHR_END}"
echo "[grits pipeline] Log file:    ${LOG_FILE}"
echo "============================================================"

# -----------------------------
# Helper: verify a step's output file is non-empty before continuing.
# Silent failures do happen — a step can exit 0 but write an empty file.
# Catching it early prevents confusing downstream errors.
# -----------------------------
check_nonempty() {
    local file="$1"
    local step="$2"
    if [[ ! -s "${file}" ]]; then
        echo "[ERROR] ${step} produced an empty or missing output file: ${file}"
        echo "[ERROR] This step sometimes silently fails with exit code 0."
        echo "[ERROR] Re-running the script from scratch often resolves it."
        exit 1
    fi
}

# ============================================================
# STEP 1: Align reads with ropebwt3 -> matches.bed
# ============================================================
MATCHES_BED="${OUT_DIR}/matches.bed"

if [[ -f "${MATCHES_BED}" ]]; then
    echo "[ropebwt3] Skipping — already exists: ${MATCHES_BED}"
else
    echo "[ropebwt3] Aligning ${SAMPLE_NAME}..."
    "${ROPEBWT3}" mem \
        -t "${THREADS}" \
        -l "${MIN_MEM_LENGTH}" \
        -p "${MAX_NUM_HITS}" \
        "${INDEX_FMD}" \
        "${FASTQ}" \
        > "${MATCHES_BED}" \
        || { echo "[ERROR] ropebwt3 failed"; exit 1; }
    echo "[ropebwt3] Done: ${MATCHES_BED}"
fi

check_nonempty "${MATCHES_BED}" "ropebwt3 (step 1)"

# ============================================================
# STEP 2: Convert BED to ps4g with PHG
# ============================================================
MATCHES_PS4G="${OUT_DIR}/matches.ps4g"

if [[ -f "${MATCHES_PS4G}" ]]; then
    echo "[phg] Skipping — already exists: ${MATCHES_PS4G}"
else
    echo "[phg] Converting BED to ps4g..."
    module load java/21 2>/dev/null || true
    "${PHG_BIN}" convert-ropebwt2ps4g-file \
        --ropebwt-bed "${MATCHES_BED}" \
        --spline-knot-dir "${SPLINES_DIR}" \
        --output-dir "${OUT_DIR}" \
        --min-mem-length "${MIN_MEM_LENGTH}" \
        --max-num-hits "${MAX_NUM_HITS}" \
        || { echo "[ERROR] phg convert failed"; exit 1; }
    echo "[phg] Done: ${MATCHES_PS4G}"
fi

check_nonempty "${MATCHES_PS4G}" "phg convert (step 2)"

# ============================================================
# STEP 3: Run GRITS impute -> out/ directory
# ============================================================
GRITS_OUT="${OUT_DIR}/out/out.bed"

if [[ -f "${GRITS_OUT}" ]]; then
    echo "[grits] Skipping — already exists: ${GRITS_OUT}"
else
    echo "[grits] Running imputation..."
    source "${CONDA_BASE}/bin/activate" "${CONDA_ENV}"
    module load java/21 2>/dev/null || true
    python "${GRITS_PY}" \
        -i "${MATCHES_PS4G}" \
        -o "${GRITS_OUT}" \
        -m seq2seq_diploid \
        -ckpt "${MODEL_CKPT}" \
        || { echo "[ERROR] GRITS impute failed"; conda deactivate 2>/dev/null; exit 1; }
    conda deactivate 2>/dev/null || true
    echo "[grits] Done: ${GRITS_OUT}"
fi

check_nonempty "${GRITS_OUT}" "GRITS impute (step 3)"

# ============================================================
# STEP 3b: Fix chromosome boundary positions in GRITS BED output
#
# WARNING: This is a known bandaid fix.
# impute.py does not correctly handle the start and end positions
# of each chromosome in its BED output. fix_bed_boundaries.py
# corrects those positions using the chr-ends file.
# This should eventually be fixed upstream in impute.py.
# ============================================================
echo "[grits] Applying chr boundary fix to: ${GRITS_OUT}"

GRITS_OUT2="${OUT_DIR}/out/out2.bed"

python /workdir/irk9/data/phg-maize/bellas_scripts/fix_bed_boundaries.py \
    "${CHR_END}" "${GRITS_OUT}" > "${GRITS_OUT2}" \
    || { echo "[ERROR] fix_bed_boundaries.py failed"; exit 1; }

check_nonempty "${GRITS_OUT2}" "fix_bed_boundaries (step 3b)"

rm "${GRITS_OUT}"
mv "${GRITS_OUT2}" "${GRITS_OUT}"

echo "[grits] Chr boundary fix applied: ${GRITS_OUT}"

# ============================================================
# STEP 4: Convert BED output to VCF
# ============================================================
OUT_VCF="${OUT_DIR}/out/out.vcf"

if [[ -f "${OUT_VCF}" ]]; then
    echo "[bed-to-vcf] Skipping — already exists: ${OUT_VCF}"
else
    echo "[bed-to-vcf] Converting to VCF..."
    module load java/21 2>/dev/null || true
    "${SAMPLE_BIN}" bed-to-vcf \
        --bed-dir "${OUT_DIR}/out" \
        --out-file "${OUT_VCF}" \
        --reference-panel-vcf "${REF_VCF}" \
        || { echo "[ERROR] bed-to-vcf failed"; exit 1; }
    echo "[bed-to-vcf] Done: ${OUT_VCF}"
fi

check_nonempty "${OUT_VCF}" "bed-to-vcf (step 4)"

echo ""
echo "All done. VCF: ${OUT_VCF}"
echo "[grits pipeline] Finished: $(date)"
