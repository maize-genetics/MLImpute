############
# DONT USE #
############


#!/usr/bin/env bash

set -uo pipefail

# ============================================================
# Run Beagle imputation for every target VCF
# Includes:
#   1. Split each target VCF by chromosome
#   2. Index split target VCFs
#   3. Run Beagle per sample/species per chromosome
#
# Requirements:
#   bcftools
#   tabix
#   beagle
# ============================================================

# -----------------------------
# User settings
# -----------------------------

# Directory containing original unsplit target VCFs:
# Example:
#   B97.vcf.gz
#   Mo17.vcf.gz
TARGET_VCF_DIR="/workdir/irk9/data/phg-maize/test2/1x/impute/cleaned_vcfs/cleaned"

# Directory where split target VCFs will be written:
TARGET_SPLIT_DIR="/workdir/irk9/data/phg-maize/test2/1x/beagle/split"

# Reference VCFs split by chromosome:
# Example:
#   ref_AF_chr1.vcf.gz
#   ref_AF_chr2.vcf.gz
REF_DIR="/workdir/irk9/data/phg-maize/test_beagle/ref_panel/vcf_per_chr"

# Beagle output directory:
OUT_DIR="/workdir/irk9/data/phg-maize/test2/1x/beagle/impute"

THREADS=10
NE=20000
SEED=-99999

CHRS=$(seq 1 10)

# Reference naming pattern:
REF_PREFIX="ref_AF"

# Set to true to overwrite existing split target VCFs
OVERWRITE_SPLIT=false

# -----------------------------
# Setup
# -----------------------------
mkdir -p "${TARGET_SPLIT_DIR}"
mkdir -p "${OUT_DIR}/logs"

echo "Starting target split + Beagle imputation"
echo "Original target VCF directory: ${TARGET_VCF_DIR}"
echo "Split target directory:        ${TARGET_SPLIT_DIR}"
echo "Reference directory:           ${REF_DIR}"
echo "Output directory:              ${OUT_DIR}"
echo

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
# Detect unsplit target VCFs
# -----------------------------
mapfile -t TARGET_VCFS < <(
    find "${TARGET_VCF_DIR}" -maxdepth 1 -type f \
        \( -name "*.vcf.gz" -o -name "*.vcf" \) \
        | sort
)

if [[ ${#TARGET_VCFS[@]} -eq 0 ]]; then
    echo "ERROR: No target VCF files found in ${TARGET_VCF_DIR}"
    echo "Expected files like: B97.vcf.gz or B97.vcf"
    exit 1
fi

echo "Found ${#TARGET_VCFS[@]} unsplit target VCF file(s)."
echo

# ============================================================
# Step 1: Split target VCFs by chromosome
# ============================================================
echo "Step 1: Splitting target VCFs by chromosome"
echo

for TARGET_VCF in "${TARGET_VCFS[@]}"; do

    FILE_NAME=$(basename "${TARGET_VCF}")

    # Remove .vcf.gz or .vcf to get sample/species name
    SAMPLE="${FILE_NAME%.vcf.gz}"
    SAMPLE="${SAMPLE%.vcf}"

    echo "Splitting target: ${SAMPLE}"

    # Index original target VCF if compressed and not already indexed
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

            bcftools view \
                -r "chr${CHR}" \
                -Oz \
                -o "${OUT_VCF}" \
                "${TARGET_VCF}"

            bcftools index "${OUT_VCF}"
        fi

    done

    echo

done

echo "Finished splitting target VCFs."
echo

# ============================================================
# Step 2: Run Beagle
# ============================================================
echo "Step 2: Running Beagle"
echo

for TARGET_VCF in "${TARGET_VCFS[@]}"; do

    FILE_NAME=$(basename "${TARGET_VCF}")
    SAMPLE="${FILE_NAME%.vcf.gz}"
    SAMPLE="${SAMPLE%.vcf}"

    SAMPLE_OUT_DIR="${OUT_DIR}/${SAMPLE}"
    mkdir -p "${SAMPLE_OUT_DIR}"

    echo "Processing sample/species with Beagle: ${SAMPLE}"

    for CHR in ${CHRS}; do

        TARGET_CHR_VCF="${TARGET_SPLIT_DIR}/${SAMPLE}_chr${CHR}.vcf.gz"
        REF_VCF="${REF_DIR}/${REF_PREFIX}_chr${CHR}.vcf.gz"
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

        beagle \
            gt="${TARGET_CHR_VCF}" \
            ref="${REF_VCF}" \
            out="${OUT_PREFIX}" \
            nthreads="${THREADS}" \
            ne="${NE}" \
            impute=true \
            seed="${SEED}" \
            2>&1 | tee "${LOG_FILE}"

    done

    echo "Finished Beagle for sample/species: ${SAMPLE}"
    echo

done

echo "All splitting and Beagle imputation jobs completed."