#!/usr/bin/env bash

set -uo pipefail

#source /workdir/irk9/software/miniconda3/bin/activate
#conda activate beagle3 

# ============================================================
# Run Beagle imputation for every target VCF
# Includes:
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

TARGET_SPLIT_DIR="$1"
REF_DIR="$2"
OUT_DIR="$3"
MAP_DIR="$4"

# ./beagle_map.sh /workdir/irk9/data/phg-maize/test2/2x/beagle/split /workdir/irk9/data/phg-maize/test_beagle/ref_panel/vcf_per_chr /workdir/irk9/data/phg-maize/test2/2x/beagle/impute_map /workdir/irk9/data/maps/maize/beagle_split

# ./beagle_map.sh /workdir/irk9/data/phg-maize/target_vcf/2x/beagle/split /workdir/irk9/data/phg-maize/test_beagle/ref_panel/vcf_per_chr /workdir/irk9/data/phg-maize/target_vcf/2x/beagle/impute_map /workdir/irk9/data/maps/maize/beagle_split

# ./beagle_map.sh /workdir/irk9/data/phg-cassava/target_vcf/2x/beagle/split /workdir/irk9/data/phg-cassava/truth-vcfs/beagle_ref/ref_split /workdir/irk9/data/phg-cassava/target_vcf/2x/beagle/impute_map /workdir/irk9/data/maps/cassava_beagle_split

# ./beagle_map.sh /workdir/irk9/data/phg-cassava/test2/2x/beagle/split /workdir/irk9/data/phg-cassava/truth-vcfs/beagle_ref/ref_split /workdir/irk9/data/phg-cassava/test2/2x/beagle/impute_map /workdir/irk9/data/maps/cassava_beagle_split

# ./beagle_map.sh /workdir/irk9/data/phg-maize/test2/5.07x/beagle/split /workdir/irk9/data/phg-maize/test_beagle/ref_panel/vcf_per_chr /workdir/irk9/data/phg-maize/test2/5.07x/beagle/impute_map /workdir/irk9/data/maps/maize/beagle_split



# Directory where split target VCFs will be written:
#TARGET_SPLIT_DIR="/workdir/irk9/data/phg-maize/test2/0.1xbeagle/split"

# Reference VCFs split by chromosome:
# Example:
#   ref_AF_chr1.vcf.gz
#   ref_AF_chr2.vcf.gz
#REF_DIR="/workdir/irk9/data/phg-maize/test_beagle/ref_panel/vcf_per_chr"

# Beagle output directory:
#OUT_DIR="/workdir/irk9/data/phg-maize/test2/0.1x/beagle/impute"

#MAP_DIR="/workdir/irk9/data/maps/maize/beagle_split"

THREADS=5
NE=20000
SEED=-99999

CHRS=$(seq 1 10)

# Reference naming pattern:
REF_PREFIX="ref_AF"

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

if [[ ! -d "${TARGET_SPLIT_DIR}" ]]; then
    echo "ERROR: Split target directory does not exist: ${TARGET_SPLIT_DIR}"
    exit 1
fi

if [[ ! -d "${REF_DIR}" ]]; then
    echo "ERROR: Reference directory does not exist: ${REF_DIR}"
    exit 1
fi

if [[ ! -d "${MAP_DIR}" ]]; then
    echo "ERROR: Map directory does not exist: ${MAP_DIR}"
    exit 1
fi

mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_DIR}/logs"

# ============================================================
# Discover samples from already-split VCFs
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
        REF_VCF="${REF_DIR}/${REF_PREFIX}_chr${CHR}_chrfiltered.vcf.gz"
        OUT_PREFIX="${SAMPLE_OUT_DIR}/${SAMPLE}_imputed_chr${CHR}"
        LOG_FILE="${OUT_DIR}/logs/${SAMPLE}_chr${CHR}.log"
        MAP="${MAP_DIR}/chr${CHR}.avgdup.map"

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
            map="${MAP}" \
            ne="${NE}" \
            impute=true \
            seed="${SEED}" \
            2>&1 | tee "${LOG_FILE}"

    done

    echo "Finished Beagle for sample/species: ${SAMPLE}"
    echo

done

echo "All splitting and Beagle imputation jobs completed."