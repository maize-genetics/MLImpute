#!/bin/bash
set -uo pipefail

TARGET_VCF_DIR=""
REF_PANEL_VCF=""
OUT_DIR=""
THREADS=5
RENAME_CHR=""

# ============================================================
# Help message
# ============================================================

usage() {
    echo
    echo "Usage:"
    echo "  $0 --step STEP --target-vcf-dir DIR --ref-panel-vcf FILE --out-dir DIR [options]"
    echo
    echo "Required arguments:"
    echo "  --target-vcf-dir DIR      Directory containing target VCF files"
    echo "  --ref-panel-vcf FILE      Reference panel VCF file"
    echo "  --out-dir DIR             Output directory"
    echo
    echo "Optional arguments:"
    echo "  --step STEP               Step to run: clean, validate_cleaning, convert_bcf, make_xcf, make_chunks, all"
    echo "                            Default: all"
    echo "  --rename-chr FILE         Chromosome rename file (required for cassava, skip for maize)"
    echo "  --map-dir DIR             Directory containing genetic maps"
    echo "  --threads INT             Number of threads"
    echo "                            Default: 5"
    echo "  -h, --help                Show this help message"
    echo
}


# ./clean.sh --target-vcf-dir /workdir/irk9/data/phg-maize/bellas_scripts/testing_suite_bella/testing_suite_test/target --ref-panel-vcf /workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz --out-dir /workdir/irk9/data/phg-maize/bellas_scripts/testing_suite_bella/testing_suite_test/target_cleaned 


# ============================================================
# Parse command-line arguments
# ============================================================

while [[ $# -gt 0 ]]
do
    case "$1" in
        --target-vcf-dir)
            TARGET_VCF_DIR="$2"
            shift 2
            ;;

        --ref-panel-vcf)
            REF_PANEL_VCF="$2"
            shift 2
            ;;

        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        
        --rename-chr)
            RENAME_CHR="$2"
            shift 2
            ;;

        --threads)
            THREADS="$2"
            shift 2
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

# ============================================================
# Check required arguments
# ============================================================

if [ -z "$TARGET_VCF_DIR" ]; then
    echo "ERROR: Missing required argument: --target-vcf-dir"
    usage
    exit 1
fi

if [ -z "$REF_PANEL_VCF" ]; then
    echo "ERROR: Missing required argument: --ref-panel-vcf"
    usage
    exit 1
fi

if [ -z "$OUT_DIR" ]; then
    echo "ERROR: Missing required argument: --out-dir"
    usage
    exit 1
fi

# ============================================================
# Validate paths
# ============================================================

if [ ! -d "$TARGET_VCF_DIR" ]; then
    echo "ERROR: Target VCF directory does not exist: $TARGET_VCF_DIR"
    exit 1
fi

if [ ! -f "$REF_PANEL_VCF" ]; then
    echo "ERROR: Reference panel VCF does not exist: $REF_PANEL_VCF"
    exit 1
fi


mkdir -p "$OUT_DIR"


LOG_DIR="$OUT_DIR/logs"

mkdir -p "$LOG_DIR"


# ============================================================
# Helper functions
# ============================================================

log_msg() {
    log_file="$1"
    msg="$2"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$log_file"
}

get_sample_name() {
    filename=$(basename "$1")

    filename=${filename%.gz}
    filename=${filename%.vcf}
    filename=${filename%_target}
    filename=${filename%_target_onlyvariants}
    filename=${filename%_target_onlyvariants_phased}
    filename=${filename%_cleaned}

    echo "$filename"
}


# ============================================================
# Clean target VCFs
# ============================================================
# Cleaning does:
#   1. fix ploidy
#   2. replace / with |
#   3. remove genotypes with missing alleles

for input_vcf in "$TARGET_VCF_DIR"/*.vcf.gz
    do
        [ -e "$input_vcf" ] || continue

        sample=$(get_sample_name "$input_vcf")
        log_file="$LOG_DIR/${sample}.cleaning.log"

        rename_chr="$OUT_DIR/${sample}.rename_chr.vcf.gz"
        fixed_vcf="$OUT_DIR/${sample}.fixploidy.vcf.gz"
        phased_vcf="$OUT_DIR/${sample}.phased.vcf.gz"
        cleaned_vcf="$OUT_DIR/${sample}.cleaned.vcf.gz"


        : > "$log_file"

        log_msg "$log_file" "Starting cleaning for sample: $sample"
        log_msg "$log_file" "Input VCF: $input_vcf"
        log_msg "$log_file" "Fixploidy output: $fixed_vcf"
        log_msg "$log_file" "Phased slash-to-pipe output: $phased_vcf"
        log_msg "$log_file" "Final cleaned output: $cleaned_vcf"


        if [ -n "$RENAME_CHR" ]; then
            bcftools annotate --rename-chrs "$RENAME_CHR" -Oz -o "$rename_chr" "$input_vcf"
            bcftools index -f "$rename_chr" >> "$log_file" 2>&1
            fixploidy_input="$rename_chr"
        else
            fixploidy_input="$input_vcf"
        fi


        log_msg "$log_file" "Running fixploidy"

        bcftools +fixploidy "$fixploidy_input" \
            -Oz \
            -o "$fixed_vcf" \
            >> "$log_file" 2>&1

        bcftools index "$fixed_vcf" >> "$log_file" 2>&1


        log_msg "$log_file" "Replacing / with | in genotype fields"

        bcftools view "$fixed_vcf" \
            | awk 'BEGIN { OFS="\t" }
                /^#/ {
                    print
                    next
                }
                {
                    for (i = 10; i <= NF; i++) {
                        n = split($i, fields, ":")
                        gsub("/", "|", fields[1])
                        $i = fields[1]
                        for (j = 2; j <= n; j++) {
                            $i = $i ":" fields[j]
                        }
                    }
                    print
                }' \
            | bgzip -c > "$phased_vcf"

        bcftools index "$phased_vcf" >> "$log_file" 2>&1


        log_msg "$log_file" "Removing records with missing genotypes"

        bcftools view \
            -e 'GT[*]="mis"' \
            -Oz \
            -o "$cleaned_vcf" \
            "$phased_vcf" \
            >> "$log_file" 2>&1

        bcftools index "$cleaned_vcf" >> "$log_file" 2>&1

        log_msg "$log_file" "Finished cleaning for sample: $sample"
done