#!/usr/bin/env bash
# Master script: clean imputed VCFs, run accuracy.py, save results.tsv
# Species: maize (cassava to be added later)
# Imputers: impute, beagle, minimac  x  map, nomap
# Coverages: 0.01, 0.1, 1, 2
#
# Truth VCF: use the full multi-sample diploid_maize.vcf.gz directly.
# AC/AN in the INFO field gives population-level allele frequencies across
# all 24 samples, which properly stratifies accuracy by MAF bin.
# accuracy.py's -s flag extracts the right sample at runtime — no per-sample
# truth VCFs needed.

set -uo pipefail   # no -e so one bad sample doesn't kill the whole run

# ============================================================
# ARGUMENT PARSING
# ============================================================

_SCRIPT_DIR_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") --base-dir DIR --truth-vcf FILE [OPTIONS]

Required:
  --base-dir   DIR   Root data directory (e.g. /workdir/irk9/data/phg-maize)
  --truth-vcf  FILE  Multi-sample truth VCF.gz

Optional:
  --target-dir DIR   Directory with imputed files [default: BASE_DIR/target_vcf]
  --script-dir DIR   Directory containing accuracy.py [default: directory of this script]
  --out-dir    DIR   Directory for outputs (results.tsv, samples_list.txt, log)
                     [default: same as --script-dir]
  --results    FILE  Output results TSV [default: OUT_DIR/results.tsv]
  --log        FILE  Log file [default: OUT_DIR/run_accuracy_TIMESTAMP.log]
  --species    STR   Species name [default: maize]
  --coverages     LIST  Comma-separated coverages [default: 0.01,0.1,1,2]
  --imputers      LIST  Comma-separated imputers [default: impute,beagle,minimac]
  --map-flags     LIST  Comma-separated map flags [default: map,nomap]
  --samples-list  FILE  File with one sample name per line (overrides auto-extract from truth VCF)
  -h, --help            Show this help and exit
EOF
}

#test 1 answer keys

#/workdir/smm477/evaluate/truth-vcfs/diploid_maize.vcf.gz
#/workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_rename_chr.vcf.gz

#test 2 answer keys

#/workdir/shared_files/cassava_test2_answer_key/cassava_diploid_filtered2.vcf.gz
#/workdir/shared_files/maize_test2_answer_key/maize_test2_filtered_fixed.vcf.gz


# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-maize --truth-vcf /workdir/smm477/evaluate/truth-vcfs/diploid_maize.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-maize/eval --imputers impute --map-flags nomap


# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-maize --truth-vcf /workdir/smm477/evaluate/truth-vcfs/diploid_maize.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-maize/eval --imputers beagle --map-flags nomap,map

# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-maize --target-dir /workdir/irk9/data/phg-maize/target_vcf --truth-vcf /workdir/smm477/evaluate/truth-vcfs/diploid_maize.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-maize/eval --imputers minimac --map-flags nomap

# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-maize --target-dir /workdir/irk9/data/phg-maize/target_vcf --truth-vcf /workdir/smm477/evaluate/truth-vcfs/diploid_maize.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-maize/eval --imputers impute --map-flags map

# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-cassava --truth-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_rename_chr.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-cassava/eval --imputers minimac --map-flags nomap

# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-cassava --target_dir  --truth-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_rename_chr.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-cassava/eval --imputers impute --map-flags map


# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-cassava --target-dir /workdir/irk9/data/phg-cassava/test2 --truth-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_rename_chr.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-cassava/eval --imputers impute,minimac --map-flags map,nomap --species cassava --samples-list /workdir/irk9/data/phg-cassava/eval/test2_samples_list.txt

# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-maize --target-dir /workdir/irk9/data/phg-maize/test2 --truth-vcf /workdir/smm477/evaluate/truth-vcfs/diploid_maize.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-maize/eval --imputers impute,minimac --map-flags map,nomap --samples-list /workdir/irk9/data/phg-maize/eval/test2_samples_list.txt

# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-cassava --truth-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_rename_chr.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-cassava/eval --species cassava --imputers minimac --map-flags map,nomap

# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-maize --truth-vcf /workdir/smm477/evaluate/truth-vcfs/diploid_maize.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-maize/eval --imputers minimac --map-flags map,nomap

# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-maize --truth-vcf /workdir/smm477/evaluate/truth-vcfs/diploid_maize.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-maize/eval --imputers beagle --map-flags map,nomap --species maize

#/workdir/irk9/data/phg-cassava/eval/test2_samples_list.txt

# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-cassava --truth-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_rename_chr.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-cassava/eval_redo --species cassava --imputers beagle --map-flags map,nomap


# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-cassava --truth-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_rename_chr.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-cassava/eval_redo --species cassava --imputers impute,minimac --map-flags map,nomap

# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-maize/ --target-dir /workdir/irk9/data/phg-maize/test2 --truth-vcf /workdir/shared_files/maize_test2_answer_key/maize_test2_filtered_fixed.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts --out-dir /workdir/irk9/data/phg-maize/test2/eval --species maize --imputers impute,minimac,beagle --map-flags nomap,map --samples-list /workdir/irk9/data/phg-maize/eval/test2_samples_list.txt

# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-cassava --target-dir /workdir/irk9/data/phg-cassava/test2 --truth-vcf /workdir/shared_files/cassava_test2_answer_key/cassava_diploid_filtered2.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts --out-dir /workdir/irk9/data/phg-cassava/test2/eval --species cassava  --imputers impute,minimac,beagle --map-flags nomap,map --samples-list /workdir/irk9/data/phg-cassava/eval/test2_samples_list.txt

# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-maize/ --target-dir /workdir/irk9/data/phg-maize/test2 --truth-vcf /workdir/shared_files/maize_test2_answer_key/maize_test2_filtered_fixed.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts --out-dir /workdir/irk9/data/phg-maize/test2/eval --species maize --imputers grits --map-flags nomap --samples-list /workdir/irk9/data/phg-maize/eval/test2_samples_list.txt


# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-cassava --target-dir /workdir/irk9/data/phg-cassava/target_vcf --truth-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_rename_chr.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-cassava/eval_redo --species cassava --imputers grits --map-flags nomap --samples-list /workdir/irk9/data/phg-cassava/target_vcf/cassava_grits_samples.txt --coverages 0.01,0.1,1,2

# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-maize/ --target-dir /workdir/irk9/data/phg-maize/target_vcf --truth-vcf /workdir/smm477/evaluate/truth-vcfs/diploid_maize.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-maize/eval --species maize --imputers grits --map-flags nomap --samples-list /workdir/irk9/data/phg-maize/target_vcf/0.01x/grits/sample_name.txt





# snp redo

# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-maize/ --target-dir /workdir/irk9/data/phg-maize/target_vcf --truth-vcf /workdir/smm477/evaluate/truth-vcfs/diploid_maize.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-maize/eval_snp  --species maize --imputers impute,minimac,beagle --map-flags map,nomap --coverages 0.01,0.1,1,2
# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-maize/ --target-dir /workdir/irk9/data/phg-maize/test2 --truth-vcf /workdir/shared_files/maize_test2_answer_key/maize_test2_filtered_fixed.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-maize/test2/eval_snp --species maize --imputers impute,minimac,beagle --map-flags map,nomap --coverages 0.01,0.1,1,2,5.07 --samples-list /workdir/irk9/data/phg-maize/eval/test2_samples_list.txt
# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-cassava --target-dir /workdir/irk9/data/phg-cassava/target_vcf --truth-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_rename_chr.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-cassava/eval_snp --species cassava --imputers impute,minimac,beagle --map-flags map,nomap --coverages 0.01,0.1,1,2
# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-cassava --target-dir /workdir/irk9/data/phg-cassava/test2_redo --truth-vcf /workdir/shared_files/cassava_test2_answer_key_redo/cassava_test2_answer_key_snps.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-cassava/test2_redo/eval_snp --species cassava --imputers impute,minimac,beagle --map-flags map,nomap --coverages 0.01,0.1,1,2,5.07,10 --samples-list /workdir/irk9/data/phg-cassava/eval/test2_samples_list.txt

# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-maize/ --target-dir /workdir/irk9/data/phg-maize/target_vcf --truth-vcf /workdir/smm477/evaluate/truth-vcfs/maize_pangenome_snps.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-maize/eval_snp  --species maize --imputers impute_haploid --map-flags map,nomap --coverages 0.01,0.1,1,2
# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-maize/ --target-dir /workdir/irk9/data/phg-maize/test2 --truth-vcf /workdir/shared_files/maize_test2_answer_key/maize_test2_haploid_snps.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-maize/test2/eval_snp --species maize --imputers impute_haploid --map-flags map,nomap --coverages 0.01,0.1,1,2,5.07 --samples-list /workdir/irk9/data/phg-maize/eval/test2_samples_list.txt

# ./run_accuracy2.sh --base-dir /workdir/irk9/data/phg-maize/ --target-dir /workdir/irk9/data/phg-maize/target_vcf --truth-vcf /workdir/smm477/evaluate/truth-vcfs/diploid_maize.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-maize/eval_snp --species maize --imputers grits --map-flags nomap --coverages 0.01,0.1,1,2 --samples-list /workdir/irk9/data/phg-maize/target_vcf/samples.txt
# ./run_accuracy2.sh --base-dir /workdir/irk9/data/phg-maize/ --target-dir /workdir/irk9/data/phg-maize/test2 --truth-vcf /workdir/shared_files/maize_test2_answer_key/maize_test2_filtered_fixed.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-maize/test2/eval_snp --species maize --imputers grits --map-flags nomap --coverages 0.01,0.1,1,2,5.07 --samples-list /workdir/irk9/data/phg-maize/eval/test2_samples_list.txt
# ./run_accuracy2.sh --base-dir /workdir/irk9/data/phg-cassava --target-dir /workdir/irk9/data/phg-cassava/target_vcf --truth-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_rename_chr.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-cassava/eval_snp --species cassava --imputers grits --map-flags nomap --coverages 0.01,0.1,1,2 --samples-list /workdir/irk9/data/phg-cassava/target_vcf/cassava_grits_samples.txt
# ./run_accuracy.sh --base-dir /workdir/irk9/data/phg-cassava --target-dir /workdir/irk9/data/phg-cassava/test2_redo --truth-vcf /workdir/shared_files/cassava_test2_answer_key_redo/cassava_test2_answer_key_snps.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-cassava/test2_redo/eval_snp --species cassava --imputers grits --map-flags nomap --coverages 0.01,0.1,5.07,26.35 --samples-list /workdir/irk9/data/phg-cassava/eval/test2_samples_list.txt

# ./run_accuracy2.sh --base-dir /workdir/irk9/data/phg-maize/ --target-dir /workdir/irk9/data/phg-maize/test3 --truth-vcf /workdir/smm477/evaluate/truth-vcfs/diploid_maize.vcf.gz --script-dir /workdir/irk9/data/phg-maize/bellas_scripts  --out-dir /workdir/irk9/data/phg-maize/test3/eval_snp --species maize --imputers grits --map-flags nomap --coverages 0.1 --samples-list /workdir/irk9/data/phg-maize/target_vcf/samples.txt

# Defaults (arrays set here; overridden by --coverages/--imputers/--map-flags below)
BASE_DIR=""
TARGET_DIR=""
TRUTH_VCF=""
SCRIPT_DIR=""
OUT_DIR=""
RESULTS_TSV=""
LOG_FILE=""
SPECIES="maize"
COVERAGES=(0.01 0.1 1 2 5.07 10 26.35)
MAP_FLAGS=(map nomap)
IMPUTERS=(impute beagle minimac grits impute_haploid)
SAMPLES_LIST_INPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-dir)   BASE_DIR="$2";                              shift 2 ;;
        --truth-vcf)  TRUTH_VCF="$2";                            shift 2 ;;
        --target-dir) TARGET_DIR="$2";                           shift 2 ;;
        --script-dir) SCRIPT_DIR="$2";                           shift 2 ;;
        --out-dir)    OUT_DIR="$2";                              shift 2 ;;
        --results)    RESULTS_TSV="$2";                          shift 2 ;;
        --log)        LOG_FILE="$2";                             shift 2 ;;
        --species)    SPECIES="$2";                              shift 2 ;;
        --coverages)  IFS=',' read -ra COVERAGES <<< "$2";       shift 2 ;;
        --imputers)      IFS=',' read -ra IMPUTERS  <<< "$2";    shift 2 ;;
        --map-flags)     IFS=',' read -ra MAP_FLAGS <<< "$2";    shift 2 ;;
        --samples-list)  SAMPLES_LIST_INPUT="$2";                shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "ERROR: Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

# Validate required args
if [[ -z "${BASE_DIR}" ]]; then
    echo "ERROR: --base-dir is required" >&2; usage; exit 1
fi
if [[ -z "${TRUTH_VCF}" ]]; then
    echo "ERROR: --truth-vcf is required" >&2; usage; exit 1
fi

# Apply derived defaults
SCRIPT_DIR="${SCRIPT_DIR:-${_SCRIPT_DIR_DEFAULT}}"
ACCURACY_PY="${SCRIPT_DIR}/accuracy.py"
TARGET_DIR="${TARGET_DIR:-${BASE_DIR}/target_vcf}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}}"
RESULTS_TSV="${RESULTS_TSV:-${OUT_DIR}/results.tsv}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/run_accuracy_$(date '+%Y%m%d_%H%M%S').log}"

# ============================================================
# LOGGING — all stdout and stderr go to terminal AND log file
# ============================================================

mkdir -p "${OUT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

log_start() {
    echo "============================================================"
    echo "[log] Run started:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "[log] base_dir:     ${BASE_DIR}"
    echo "[log] target_dir:   ${TARGET_DIR}"
    echo "[log] truth_vcf:    ${TRUTH_VCF}"
    echo "[log] script_dir:   ${SCRIPT_DIR}"
    echo "[log] out_dir:      ${OUT_DIR}"
    echo "[log] results_tsv:  ${RESULTS_TSV}"
    echo "[log] log_file:     ${LOG_FILE}"
    echo "[log] species:      ${SPECIES}"
    echo "[log] coverages:    ${COVERAGES[*]}"
    echo "[log] imputers:     ${IMPUTERS[*]}"
    echo "[log] map_flags:    ${MAP_FLAGS[*]}"
    echo "============================================================"
}
log_start

trap 'echo "[log] Run finished: $(date '"'"'+%Y-%m-%d %H:%M:%S'"'"') (exit $?)"' EXIT

# ============================================================
# CONFIGURATION (derived above; kept here for visibility)
# ============================================================

# Species-specific configuration: chromosomes and minimac directory/file-pattern naming.
# Minimac dir and file-middle (FMID) differ per species AND map flag:
#   maize   map   -> minimac_map      / .cleaned_imputed
#   maize   nomap -> minimac_redo     / _target_imputed
#   cassava map   -> minimac_map_redo / .rename_chr_imputed
#   cassava nomap -> minimac_redo     / .rename_chr_imputed
case "${SPECIES}" in
    maize)
        CHROMS=(1 2 3 4 5 6 7 8 9 10)
        MINIMAC_MAP_DIR="minimac_map"
        MINIMAC_NOMAP_DIR="minimac"
        MINIMAC_MAP_FMID=".cleaned_imputed"
        MINIMAC_NOMAP_FMID="_target_imputed"
        ;;
    cassava)
        CHROMS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
        MINIMAC_MAP_DIR="minimac_map"
        MINIMAC_NOMAP_DIR="minimac"
        MINIMAC_MAP_FMID=".rename_chr_imputed"
        MINIMAC_NOMAP_FMID=".rename_chr_imputed"
        ;;
    *)
        echo "ERROR: Unknown species '${SPECIES}'. Supported: maize, cassava" >&2; exit 1 ;;
esac

# Genetic map directory naming convention (impute and beagle; minimac handled above):
#   map   -> {imputer}_map  (e.g. impute_map, beagle_map)
#   nomap -> {imputer}      (e.g. impute, beagle)
model_dir() {
    local imputer=$1 map_flag=$2
    if [[ "${map_flag}" == "map" ]]; then
        echo "${imputer}_map"
    else
        echo "${imputer}"
    fi
}

# ============================================================
# STEP 1: Get sample list from truth VCF (runs once)
# ============================================================
get_samples() {
    local samples_list="${OUT_DIR}/samples_list.txt"
    if [[ -n "${SAMPLES_LIST_INPUT}" ]]; then
        echo "[setup] Using provided samples list: ${SAMPLES_LIST_INPUT}"
        cp "${SAMPLES_LIST_INPUT}" "${samples_list}"
    else
        echo "[setup] Extracting sample names from truth VCF..."
        bcftools query -l "${TRUTH_VCF}" > "${samples_list}"
    fi
    echo "[setup] Found $(wc -l < "${samples_list}") samples"
}

# ============================================================
# STEP 2: Clean functions — concat + reheader per imputer type
# Returns path in IMPUTED_VCF, or empty string on skip/error
# ============================================================
IMPUTED_VCF=""

clean_impute() {
    local sample=$1 coverage=$2 map_flag=$3 imputer_name=${4:-impute}
    IMPUTED_VCF=""
    local mdir
    mdir=$(model_dir "${imputer_name}" "${map_flag}")

    # Chunk files live in:
    #   nomap: {TARGET_DIR}/{coverage}x/{mdir}/imputed/{sample}.cleaned.target.bcf/
    #   map:   {TARGET_DIR}/{coverage}x/{mdir}/{sample}.cleaned.target.bcf/
    local chunk_dir
    if [[ "${imputer_name}" == "impute" && "${map_flag}" == "map" ]]; then
        chunk_dir="${TARGET_DIR}/${coverage}x/${mdir}/${sample}.cleaned.target.bcf"
    elif [[ "${imputer_name}" == "impute_haploid" && "${map_flag}" == "map" ]]; then
        chunk_dir="${TARGET_DIR}/${coverage}x/${mdir}/${sample}.cleaned"
    elif [[ "${imputer_name}" == "impute_haploid" && "${map_flag}" == "nomap" ]]; then
        chunk_dir="${TARGET_DIR}/${coverage}x/${mdir}/${sample}.cleaned.target.bcf"
    else
        chunk_dir="${TARGET_DIR}/${coverage}x/${mdir}/imputed/${sample}.cleaned.target.bcf"
    fi

    local work_dir="${TARGET_DIR}/${coverage}x/${mdir}"
    local vcf_list="${work_dir}/vcf_list_${sample}.txt"
    local concat_vcf="${work_dir}/${sample}_concat.vcf"
    local final_vcf="${work_dir}/${sample}_${coverage}_${map_flag}_${imputer_name}.vcf.gz"

    if [[ ! -d "${chunk_dir}" ]]; then
        echo "[skip ${imputer_name}] Missing dir: ${chunk_dir}"
        return 0
    fi

    if [[ -f "${final_vcf}" ]]; then
        echo "[${imputer_name}] Already cleaned: ${final_vcf}"
        IMPUTED_VCF="${final_vcf}"
        return 0
    fi

    # Build sorted file list (ascending chr then chunk)
    > "${vcf_list}_unsorted"
    for f in "${chunk_dir}"/*.imputed.bcf; do
        fname=$(basename "$f")
        if [[ "$fname" =~ chr([0-9]+)\.chunk([0-9]+) ]]; then
            echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]} $f" >> "${vcf_list}_unsorted"
        fi
    done
    sort -k1,1n -k2,2n "${vcf_list}_unsorted" | awk '{print $3}' > "${vcf_list}"
    rm -f "${vcf_list}_unsorted"

    if [[ ! -s "${vcf_list}" ]]; then
        echo "[skip ${imputer_name}] No chunk files in ${chunk_dir}"
        return 0
    fi

    echo "[${imputer_name}] Concatenating $(wc -l < "${vcf_list}") chunks for ${sample} cov=${coverage} ${map_flag}"
    bcftools concat -f "${vcf_list}" -Ov -o "${concat_vcf}" || { echo "[WARN] concat failed"; return 0; }

    echo "${sample}" > "${work_dir}/_sname.txt"
    bcftools reheader -s "${work_dir}/_sname.txt" -o "${work_dir}/_renamed.vcf" "${concat_vcf}" \
        && bgzip -f "${work_dir}/_renamed.vcf" \
        && mv "${work_dir}/_renamed.vcf.gz" "${final_vcf}" \
        && bcftools index -t "${final_vcf}" \
        || { echo "[WARN] reheader/bgzip failed for ${imputer_name} ${sample}"; return 0; }

    rm -f "${concat_vcf}" "${vcf_list}" "${work_dir}/_sname.txt"
    echo "[${imputer_name}] Done: ${final_vcf}"
    IMPUTED_VCF="${final_vcf}"
}

clean_grits() {
    local sample=$1 coverage=$2 map_flag=$3
    IMPUTED_VCF=""

    # grits has no map option; skip the map iteration to avoid duplicate results
    if [[ "${map_flag}" != "nomap" ]]; then
        return 0
    fi

    local work_dir="${TARGET_DIR}/${coverage}x/grits/out"
    local src_vcf="${work_dir}/out.vcf"
    local final_vcf="${work_dir}/${sample}_${coverage}_grits.vcf.gz"

    if [[ ! -f "${src_vcf}" ]]; then
        echo "[skip grits] Missing: ${src_vcf}"
        return 0
    fi

    if [[ -f "${final_vcf}" ]]; then
        echo "[grits] Already cleaned: ${final_vcf}"
        IMPUTED_VCF="${final_vcf}"
        return 0
    fi

    echo "[grits] Renaming sample 'out' -> '${sample}' for cov=${coverage}"
    echo "${sample}" > "${work_dir}/_sname.txt"

    local contig_hdr="${work_dir}/_contig_hdr.txt"
    local tmp_vcf="${work_dir}/_tmp_${sample}.vcf"
    # Declare Chromosome01..18 contigs (not chr1..18) — rename-chrs will convert them
    {
        for i in $(seq 1 10); do
            printf '##contig=<ID=chr%01d>\n' "$i"
        done
        echo '##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in genotypes">'
        echo '##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles in called genotypes">'
    } > "${contig_hdr}"

    # Add Chromosome contig declarations to header before reheader, which fails
    # if the CHROM values are not defined in the ##contig lines
    bcftools annotate -h "${contig_hdr}" -Ov -o "${tmp_vcf}" "${src_vcf}" \
        || { echo "[WARN] pre-annotate failed for grits ${sample}"; rm -f "${tmp_vcf}" "${contig_hdr}"; return 0; }

    bcftools reheader -s "${work_dir}/_sname.txt" "${tmp_vcf}" \
        | bgzip -c > "${final_vcf}" \
        && bcftools index -t "${final_vcf}" \
        || { echo "[WARN] reheader/bgzip failed for grits ${sample}"; rm -f "${final_vcf}" "${tmp_vcf}"; return 0; }

    rm -f "${work_dir}/_sname.txt" "${contig_hdr}" "${tmp_vcf}"
    echo "[grits] Done: ${final_vcf}"
    IMPUTED_VCF="${final_vcf}"
}

clean_minimac() {
    local sample=$1 coverage=$2 map_flag=$3
    IMPUTED_VCF=""

    local mdir fmid
    if [[ "${map_flag}" == "map" ]]; then
        mdir="${MINIMAC_MAP_DIR}"; fmid="${MINIMAC_MAP_FMID}"
    else
        mdir="${MINIMAC_NOMAP_DIR}"; fmid="${MINIMAC_NOMAP_FMID}"
    fi

    local work_dir="${TARGET_DIR}/${coverage}x/${mdir}"
    local final_vcf="${work_dir}/${sample}_${coverage}_${map_flag}_minimac.vcf.gz"

    if [[ ! -d "${work_dir}" ]]; then
        echo "[skip minimac] Missing dir: ${work_dir}"
        return 0
    fi

    if [[ -f "${final_vcf}" ]]; then
        echo "[minimac] Already cleaned: ${final_vcf}"
        IMPUTED_VCF="${final_vcf}"
        return 0
    fi

    local vcf_list="${work_dir}/vcf_list_${sample}.txt"
    > "${vcf_list}"
    for chr in "${CHROMS[@]}"; do
        local f="${work_dir}/${sample}${fmid}_chr${chr}.vcf.gz"
        [[ -f "$f" ]] && echo "$f" >> "${vcf_list}"
    done

    if [[ ! -s "${vcf_list}" ]]; then
        echo "[skip minimac] No chr files found for ${sample} in ${work_dir}"
        return 0
    fi

    echo "[minimac] Concatenating $(wc -l < "${vcf_list}") chromosomes for ${sample} cov=${coverage} ${map_flag}"
    local concat_vcf="${work_dir}/${sample}_concat.vcf.gz"
    bcftools concat -f "${vcf_list}" -Oz -o "${concat_vcf}" \
        || { echo "[WARN] concat failed"; return 0; }

    # Rename sample: minimac names the sample after the input BCF path
    echo "${sample}" > "${work_dir}/_sname.txt"
    bcftools reheader -s "${work_dir}/_sname.txt" -o "${final_vcf}" "${concat_vcf}" \
        && bcftools index -t "${final_vcf}" \
        || { echo "[WARN] reheader failed for minimac ${sample}"; rm -f "${concat_vcf}"; return 0; }

    rm -f "${concat_vcf}" "${vcf_list}" "${work_dir}/_sname.txt"
    echo "[minimac] Done: ${final_vcf}"
    IMPUTED_VCF="${final_vcf}"
}

clean_beagle() {
    local sample=$1 coverage=$2 map_flag=$3
    IMPUTED_VCF=""

    # Beagle layout: beagle/impute_map/ (map) or beagle/impute/ (nomap)
    local impute_subdir
    if [[ "${map_flag}" == "map" ]]; then
        impute_subdir="impute_map"
    else
        impute_subdir="impute"
    fi

    local work_dir="${TARGET_DIR}/${coverage}x/beagle"
    local chr_dir="${work_dir}/${impute_subdir}/${sample}.cleaned"
    local final_vcf="${work_dir}/${sample}_${coverage}_${map_flag}_beagle.vcf.gz"

    if [[ ! -d "${chr_dir}" ]]; then
        echo "[skip beagle] Missing dir: ${chr_dir}"
        return 0
    fi

    if [[ -f "${final_vcf}" ]]; then
        echo "[beagle] Already cleaned: ${final_vcf}"
        IMPUTED_VCF="${final_vcf}"
        return 0
    fi

    local vcf_list="${work_dir}/vcf_list_${sample}_${map_flag}.txt"
    > "${vcf_list}"
    for chr in "${CHROMS[@]}"; do
        local f="${chr_dir}/${sample}.cleaned_imputed_chr${chr}.vcf.gz"
        if [[ -f "$f" ]]; then
            if [[ ! -f "${f}.tbi" ]]; then
                if ! bcftools index -t "$f" 2>/dev/null; then
                    echo "[beagle] VCF unsorted, sorting: $(basename "$f")"
                    local tmp_sorted="${f%.vcf.gz}.sorted.vcf.gz"
                    bcftools annotate -h <(echo "##contig=<ID=chr${chr}>") "$f" \
                        | bcftools sort -Oz -o "${tmp_sorted}" \
                        && mv "${tmp_sorted}" "$f" \
                        && bcftools index -t "$f" \
                        || { echo "[WARN] sort/index failed for $(basename "$f")"; continue; }
                fi
            fi
            echo "$f" >> "${vcf_list}"
        fi
    done

    if [[ ! -s "${vcf_list}" ]]; then
        echo "[skip beagle] No chr files found for ${sample} in ${chr_dir}"
        return 0
    fi

    echo "[beagle] Concatenating $(wc -l < "${vcf_list}") chromosomes for ${sample} cov=${coverage} ${map_flag}"
    local concat_vcf="${work_dir}/${sample}_${map_flag}_concat.vcf.gz"
    bcftools concat -f "${vcf_list}" -Oz -o "${concat_vcf}" || { echo "[WARN] concat failed"; return 0; }

    echo "${sample}" > "${work_dir}/_sname.txt"
    bcftools reheader -s "${work_dir}/_sname.txt" -o "${final_vcf}" "${concat_vcf}" \
        && bcftools index -t "${final_vcf}" \
        || { echo "[WARN] reheader failed for beagle ${sample}"; return 0; }

    rm -f "${concat_vcf}" "${vcf_list}" "${work_dir}/_sname.txt"
    echo "[beagle] Done: ${final_vcf}"
    IMPUTED_VCF="${final_vcf}"
}

# ============================================================
# STEP 3: Run accuracy.py and append one row to results.tsv
# ============================================================
run_accuracy() {
    local species=$1 sample=$2 coverage=$3 map_flag=$4 imputer=$5 imputed_vcf=$6
    local truth_vcf="${TRUTH_VCF}"

    echo "[accuracy] Running: ${sample} cov=${coverage} ${map_flag} ${imputer}"
    local raw
    raw=$(python "${ACCURACY_PY}" \
        --truth "${truth_vcf}" \
        --imputed "${imputed_vcf}" \
        -s "${sample}" \
        --partial-credit \
        2>/dev/null) || { echo "[WARN] accuracy.py failed for ${imputer} ${sample}"; return 0; }

    # --- Save full accuracy.py output with input parameters ---
    local acc_dir="${OUT_DIR}/accuracy_outputs"
    mkdir -p "${acc_dir}"
    {
        echo "# species:   ${species}"
        echo "# sample:    ${sample}"
        echo "# coverage:  ${coverage}"
        echo "# map_flag:  ${map_flag}"
        echo "# imputer:   ${imputer}"
        echo "# truth_vcf: ${truth_vcf}"
        echo "# imputed:   ${imputed_vcf}"
        echo "# run_date:  $(date '+%Y-%m-%d %H:%M:%S')"
        echo "#"
        echo "${raw}"
    } > "${acc_dir}/${species}_${sample}_${coverage}_${map_flag}_${imputer}.txt"

    # --- Parse summary block ---
    local compared_sites missing_sites allele_gt partial_allele
    compared_sites=$(echo "${raw}"  | awk '/compared_sites/{print $NF}')
    missing_sites=$(echo "${raw}"   | awk '/missing_in_imputed_sites/{print $NF}')
    allele_gt=$(echo "${raw}"       | grep 'allele_GT_concordance' | grep -v partial | awk '{print $2}')
    partial_allele=$(echo "${raw}"  | awk '/partial_allele_concordance/{print $NF}')

    # --- Overall R2 = mean of all non-NA per-AF-bin R2 values ---
    local r2_overall
    r2_overall=$(echo "${raw}" | awk '
        /R2 OF ALL/{in_r2=1; next}
        in_r2 && /^[0-9]/ && $3 != "None" && $3+0 == $3 { sum += $3; n++ }
        END { if (n > 0) print sum/n; else print "NA" }
    ')
    r2_overall="${r2_overall:-NA}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${species}" "${sample}" "${coverage}" "${map_flag}" "${imputer}" \
        "${compared_sites:-NA}" "${missing_sites:-NA}" \
        "${allele_gt:-NA}" "${partial_allele:-NA}" "${r2_overall}" \
        >> "${RESULTS_TSV}"

    echo "[done] ${sample} ${coverage} ${map_flag} ${imputer} | GT=${allele_gt:-NA} partial=${partial_allele:-NA} R2=${r2_overall}"
}

# ============================================================
# MAIN
# ============================================================

# Check if results.tsv exists and has a header already
# Columns: species, sample, coverage, map, imputer,
#          compared_sites, missing_in_imputed_sites,
#          allele_GT_concordance, partial_allele_concordance, R2_overall,
#          hom_acc_bins (20 tab-sep values), het_acc_bins (11 tab-sep values)
if [[ ! -f "${RESULTS_TSV}" ]]; then
    {
        printf 'species\tsample\tcoverage\tmap\timputer\tcompared_sites\tmissing_in_imputed_sites\tallele_GT_concordance\tpartial_allele_concordance\tR2_overall\n'
    } > "${RESULTS_TSV}"
fi

# Get sample list
get_samples

# Loop over all combinations
while IFS= read -r sample; do
    for coverage in "${COVERAGES[@]}"; do
        for map_flag in "${MAP_FLAGS[@]}"; do
            for imputer in "${IMPUTERS[@]}"; do

                # Skip if this combination is already in results.tsv
                if grep -q "^${SPECIES}	${sample}	${coverage}	${map_flag}	${imputer}	" "${RESULTS_TSV}" 2>/dev/null; then
                    echo "[skip] Already in results: ${sample} ${coverage} ${map_flag} ${imputer}"
                    continue
                fi

                echo ""
                echo "=== ${SPECIES} | ${sample} | cov=${coverage} | ${map_flag} | ${imputer} ==="

                case "${imputer}" in
                    impute)         clean_impute   "${sample}" "${coverage}" "${map_flag}" ;;
                    impute_haploid) clean_impute   "${sample}" "${coverage}" "${map_flag}" "impute_haploid" ;;
                    minimac)        clean_minimac  "${sample}" "${coverage}" "${map_flag}" ;;
                    beagle)         clean_beagle   "${sample}" "${coverage}" "${map_flag}" ;;
                    grits)          clean_grits    "${sample}" "${coverage}" "${map_flag}" ;;
                esac

                if [[ -n "${IMPUTED_VCF}" ]]; then
                    snp_vcf="${IMPUTED_VCF%.vcf.gz}_snps.vcf.gz"
                    if [[ ! -f "${snp_vcf}" ]]; then
                        echo "[snp-filter] Filtering SNPs: ${sample} cov=${coverage} ${map_flag} ${imputer}"
                        bcftools view -v snps "${IMPUTED_VCF}" -Oz -o "${snp_vcf}" \
                            && bcftools index -t "${snp_vcf}" \
                            || { echo "[WARN] SNP filter failed for ${imputer} ${sample}"; snp_vcf="${IMPUTED_VCF}"; }
                    fi
                    run_accuracy "${SPECIES}" "${sample}" "${coverage}" "${map_flag}" "${imputer}" "${snp_vcf}"
                fi

            done
        done
    done
done < "${OUT_DIR}/samples_list.txt"

echo ""
echo "All done. Results: ${RESULTS_TSV}"
