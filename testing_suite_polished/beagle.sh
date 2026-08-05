#!/usr/bin/env bash

set -uo pipefail

# ============================================================
# Run Beagle imputation
# Supports maize (chr1-10) and cassava (chr1-18)
# Optionally splits target VCFs before imputation
# Optionally uses a genetic map
#
# Requirements:
#   bcftools, tabix, beagle
#
# Usage:
#   beagle.sh --target-split-dir DIR --ref-dir DIR --out-dir DIR [options]
#
# Required:
#   --target-split-dir DIR   Directory with (or for) per-chromosome split VCFs
#   --ref-dir DIR            Directory with reference VCFs per chromosome
#   --out-dir DIR            Beagle output directory
#
# Optional:
#   --target-vcf-dir DIR     Unsplit VCF directory; if given, splits into --target-split-dir
#   --map-dir DIR            Genetic map directory; enables map= in Beagle
#   --species maize|cassava  Species (default: maize; sets chr count and file naming)
#   --threads N              Number of threads (default: 5)
#   --ne N                   Effective population size (default: 20000)
#   --seed N                 Random seed (default: -99999)
#   --conda-path PATH        Path to conda activate script
#   --conda-env NAME         Conda environment name (default: beagle3)
#   --ref-suffix SUFFIX      Override reference VCF suffix (e.g. _chrfiltered.vcf.gz)
#   --map-ext EXT            Override map file extension (e.g. avgdup.map)
#   --overwrite-split        Overwrite existing split VCFs
# ============================================================

# -----------------------------
# Defaults
# -----------------------------
TARGET_SPLIT_DIR=""
TARGET_VCF_DIR=""
REF_DIR=""
OUT_DIR=""
MAP_DIR=""
SPECIES="maize"
THREADS=5
NE=20000
SEED=-99999
CONDA_PATH="/workdir/irk9/software/miniconda3/bin/activate"
CONDA_ENV="beagle3"
REF_SUFFIX=""
MAP_EXT=""
OVERWRITE_SPLIT=false

# ./beagle.sh --target-vcf-dir /workdir/irk9/data/phg-maize/bellas_scripts/testing_suite_bella/testing_suite_test/target_cleaned --target-split-dir /workdir/irk9/data/phg-maize/bellas_scripts/testing_suite_bella/testing_suite_test/beagle/split --ref-dir /workdir/irk9/data/phg-maize/test_beagle/ref_panel/vcf_per_chr --out-dir  /workdir/irk9/data/phg-maize/bellas_scripts/testing_suite_bella/testing_suite_test/beagle/impute --species maize



# -----------------------------
# Parse arguments
# -----------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-split-dir) TARGET_SPLIT_DIR="$2"; shift 2 ;;
        --target-vcf-dir)   TARGET_VCF_DIR="$2";   shift 2 ;;
        --ref-dir)          REF_DIR="$2";           shift 2 ;;
        --out-dir)          OUT_DIR="$2";           shift 2 ;;
        --map-dir)          MAP_DIR="$2";           shift 2 ;;
        --species)          SPECIES="$2";           shift 2 ;;
        --threads)          THREADS="$2";           shift 2 ;;
        --ne)               NE="$2";               shift 2 ;;
        --seed)             SEED="$2";             shift 2 ;;
        --conda-path)       CONDA_PATH="$2";       shift 2 ;;
        --conda-env)        CONDA_ENV="$2";        shift 2 ;;
        --ref-suffix)       REF_SUFFIX="$2";       shift 2 ;;
        --map-ext)          MAP_EXT="$2";          shift 2 ;;
        --overwrite-split)  OVERWRITE_SPLIT=true;  shift 1 ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

# -----------------------------
# Validate required arguments
# -----------------------------
ERRORS=0
if [[ -z "${TARGET_SPLIT_DIR}" ]]; then
    echo "ERROR: Missing required argument: --target-split-dir"
    ERRORS=$((ERRORS + 1))
fi
if [[ -z "${REF_DIR}" ]]; then
    echo "ERROR: Missing required argument: --ref-dir"
    ERRORS=$((ERRORS + 1))
fi
if [[ -z "${OUT_DIR}" ]]; then
    echo "ERROR: Missing required argument: --out-dir"
    ERRORS=$((ERRORS + 1))
fi
if [[ $ERRORS -gt 0 ]]; then
    echo ""
    echo "Usage: $(basename "$0") --target-split-dir DIR --ref-dir DIR --out-dir DIR [options]"
    echo "Run '$(basename "$0") --help' or see the header comments for full option list."
    exit 1
fi

# -----------------------------
# Species-specific defaults
# -----------------------------
case "${SPECIES}" in
    maize)
        CHRS=$(seq 1 10)
        [[ -z "${REF_SUFFIX}" ]] && REF_SUFFIX="_chrfiltered.vcf.gz"
        [[ -z "${MAP_EXT}" ]]    && MAP_EXT="avgdup.map"
        ;;
    cassava)
        CHRS=$(seq 1 18)
        [[ -z "${REF_SUFFIX}" ]] && REF_SUFFIX="_renamed.vcf.gz"
        [[ -z "${MAP_EXT}" ]]    && MAP_EXT="map"
        ;;
    *)
        echo "ERROR: Unknown species '${SPECIES}'. Expected: maize or cassava"
        exit 1
        ;;
esac

REF_PREFIX="ref_AF"

# -----------------------------
# Activate conda environment
# -----------------------------
source "${CONDA_PATH}" "${CONDA_ENV}"

# -----------------------------
# Check required programs
# -----------------------------
for PROGRAM in bcftools tabix beagle; do
    if ! command -v "${PROGRAM}" >/dev/null 2>&1; then
        echo "ERROR: Required program not found in PATH: ${PROGRAM}"
        exit 1
    fi
done

# -----------------------------
# Check directories
# -----------------------------
if [[ -n "${TARGET_VCF_DIR}" && ! -d "${TARGET_VCF_DIR}" ]]; then
    echo "ERROR: Target VCF directory does not exist: ${TARGET_VCF_DIR}"
    exit 1
fi

if [[ ! -d "${REF_DIR}" ]]; then
    echo "ERROR: Reference directory does not exist: ${REF_DIR}"
    exit 1
fi

if [[ -n "${MAP_DIR}" && ! -d "${MAP_DIR}" ]]; then
    echo "ERROR: Map directory does not exist: ${MAP_DIR}"
    exit 1
fi

mkdir -p "${TARGET_SPLIT_DIR}"
mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_DIR}/logs"

# -----------------------------
# Run-level summary log
# -----------------------------
RUN_LOG="${OUT_DIR}/logs/run_summary.log"
{
    echo "===== Beagle run started: $(date) ====="
    echo "  species:          ${SPECIES}"
    echo "  target-split-dir: ${TARGET_SPLIT_DIR}"
    echo "  target-vcf-dir:   ${TARGET_VCF_DIR:-<not provided>}"
    echo "  ref-dir:          ${REF_DIR}"
    echo "  out-dir:          ${OUT_DIR}"
    echo "  map-dir:          ${MAP_DIR:-<not provided>}"
    echo "  threads:          ${THREADS}"
    echo "  ne:               ${NE}"
    echo "  seed:             ${SEED}"
    echo "  ref-suffix:       ${REF_SUFFIX}"
    echo "  map-ext:          ${MAP_EXT}"
    echo "  chromosomes:      $(echo "${CHRS}" | tr '\n' ' ')"
    echo ""
} | tee "${RUN_LOG}"

# ============================================================
# Step 1: Split target VCFs (only if --target-vcf-dir given)
# ============================================================
if [[ -n "${TARGET_VCF_DIR}" ]]; then
    echo "Step 1: Splitting target VCFs by chromosome"
    echo

    mapfile -t TARGET_VCFS < <(
        find "${TARGET_VCF_DIR}" -maxdepth 1 -type f \
            \( -name "*.vcf.gz" -o -name "*.vcf" \) \
            | sort
    )

    if [[ ${#TARGET_VCFS[@]} -eq 0 ]]; then
        echo "ERROR: No target VCF files found in ${TARGET_VCF_DIR}"
        exit 1
    fi

    echo "Found ${#TARGET_VCFS[@]} unsplit target VCF file(s)."
    echo

    for TARGET_VCF in "${TARGET_VCFS[@]}"; do
        FILE_NAME=$(basename "${TARGET_VCF}")
        SAMPLE="${FILE_NAME%.vcf.gz}"
        SAMPLE="${SAMPLE%.vcf}"

        echo "Splitting target: ${SAMPLE}"

        if [[ "${TARGET_VCF}" == *.vcf.gz ]]; then
            if [[ ! -f "${TARGET_VCF}.tbi" && ! -f "${TARGET_VCF}.csi" ]]; then
                echo "  Indexing original VCF: ${TARGET_VCF}"
                tabix -p vcf "${TARGET_VCF}"
            fi
        fi

        for CHR in ${CHRS}; do
            OUT_VCF="${TARGET_SPLIT_DIR}/${SAMPLE}_chr${CHR}.vcf.gz"

            if [[ -f "${OUT_VCF}" && "${OVERWRITE_SPLIT}" != "true" ]]; then
                echo "  chr${CHR}: split VCF already exists, skipping"
            else
                echo "  chr${CHR}: creating ${OUT_VCF}"
                bcftools view -r "chr${CHR}" -Oz -o "${OUT_VCF}" "${TARGET_VCF}"
                bcftools index "${OUT_VCF}"
            fi
        done
        echo
    done

    echo "Finished splitting target VCFs."
    echo
fi

# ============================================================
# Discover samples from split VCFs
# ============================================================
echo "Discovering samples from split VCF directory:"
echo "  ${TARGET_SPLIT_DIR}"
echo

mapfile -t SAMPLES < <(
    find "${TARGET_SPLIT_DIR}" -maxdepth 1 -type f -name "*_chr*.vcf.gz" \
        -printf "%f\n" \
        | sed -E 's/_chr[0-9]+\.vcf\.gz$//' \
        | sort -u
)

if [[ "${#SAMPLES[@]}" -eq 0 ]]; then
    echo "ERROR: No split target VCFs found in ${TARGET_SPLIT_DIR}"
    echo "Expected files like: SAMPLE_chr1.vcf.gz"
    exit 1
fi

echo "Found ${#SAMPLES[@]} sample/species name(s):"
printf "  %s\n" "${SAMPLES[@]}"
echo

# ============================================================
# Run Beagle
# ============================================================
echo "Running Beagle"
echo

for SAMPLE in "${SAMPLES[@]}"; do
    SAMPLE_OUT_DIR="${OUT_DIR}/${SAMPLE}"
    mkdir -p "${SAMPLE_OUT_DIR}"

    echo "Processing sample/species with Beagle: ${SAMPLE}"

    for CHR in ${CHRS}; do
        TARGET_CHR_VCF="${TARGET_SPLIT_DIR}/${SAMPLE}_chr${CHR}.vcf.gz"
        REF_VCF="${REF_DIR}/${REF_PREFIX}_chr${CHR}${REF_SUFFIX}"
        OUT_PREFIX="${SAMPLE_OUT_DIR}/${SAMPLE}_imputed_chr${CHR}"
        LOG_FILE="${OUT_DIR}/logs/${SAMPLE}_chr${CHR}.log"

        if [[ ! -f "${TARGET_CHR_VCF}" ]]; then
            echo "WARNING: Missing split target VCF, skipping: ${TARGET_CHR_VCF}"
            continue
        fi

        if [[ ! -f "${REF_VCF}" ]]; then
            echo "WARNING: Missing reference VCF, skipping: ${REF_VCF}"
            continue
        fi

        echo "  Running chr${CHR}"
        echo "    target: ${TARGET_CHR_VCF}"
        echo "    ref:    ${REF_VCF}"
        echo "    out:    ${OUT_PREFIX}"

        if [[ -n "${MAP_DIR}" ]]; then
            MAP="${MAP_DIR}/chr${CHR}.${MAP_EXT}"
            beagle \
                gt="${TARGET_CHR_VCF}" \
                ref="${REF_VCF}" \
                out="${OUT_PREFIX}" \
                nthreads="${THREADS}" \
                map="${MAP}" \
                ne="${NE}" \
                impute=true \
                seed="${SEED}" \
                2>&1 | tee "${LOG_FILE}"
        else
            beagle \
                gt="${TARGET_CHR_VCF}" \
                ref="${REF_VCF}" \
                out="${OUT_PREFIX}" \
                nthreads="${THREADS}" \
                ne="${NE}" \
                impute=true \
                seed="${SEED}" \
                2>&1 | tee "${LOG_FILE}"
        fi

    done

    echo "Finished Beagle for sample/species: ${SAMPLE}"
    echo
done

echo "All Beagle imputation jobs completed." | tee -a "${RUN_LOG}"
