#!/bin/bash
set -uo pipefail

# REMINDER: make sure to retrieve the path to impute functions

export PATH=/programs/impute5_v1.2.0:$PATH


# ============================================================
# Default settings
# ============================================================

TARGET_VCF_DIR=""
OUT_DIR=""
MAP_DIR="/workdir/irk9/data/maps/split_by_pos_cleaned"
THREADS=5
# ./clean_chunk_xcftools_impute_w_map.sh --target-vcf-dir /workdir/irk9/data/phg-cassava/target_vcf/0.01x/impute --out-dir /workdir/irk9/data/phg-cassava/target_vcf/0.01x/impute_map --map-dir /workdir/irk9/data/maps/split_by_pos_cleaned_no_neg --threads 5

# ./clean_chunk_xcftools_impute_w_map.sh --target-vcf-dir /workdir/irk9/data/phg-cassava/test2/2x/impute --out-dir /workdir/irk9/data/phg-cassava/test2/2x/impute_map --map-dir /workdir/irk9/data/maps/split_by_pos_cleaned_no_neg --threads 5

# ./clean_chunk_xcftools_impute_w_map.sh --target-vcf-dir /workdir/irk9/data/phg-cassava/test2/5.07x/impute --out-dir /workdir/irk9/data/phg-cassava/test2/5.07x/impute_map --map-dir /workdir/irk9/data/maps/split_by_pos_cleaned_no_neg --threads 5


# cassava test 2 redo

# ./clean_chunk_xcftools_impute_w_map.sh --target-vcf-dir /workdir/irk9/data/phg-cassava/test2_redo/26.35x/impute --out-dir /workdir/irk9/data/phg-cassava/test2_redo/26.35x/impute_map --map-dir /workdir/irk9/data/maps/split_by_pos_cleaned_no_neg --threads 20

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

        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;

        --map-dir)
            MAP_DIR="$2"
            shift 2
            ;;

        --threads)
            THREADS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ============================================================
# Check required arguments
# ============================================================

if [ -z "$TARGET_VCF_DIR" ]; then
    echo "ERROR: Missing required argument: --target-vcf-dir"
    exit 1
fi

if [ -z "$MAP_DIR" ]; then
    echo "ERROR: Missing required argument: --map-dir"
    exit 1
fi

if [ -z "$OUT_DIR" ]; then
    echo "ERROR: Missing required argument: --out-dir"
    exit 1
fi

# ============================================================
# Validate paths
# ============================================================

if [ ! -d "$TARGET_VCF_DIR" ]; then
    echo "ERROR: Target VCF directory does not exist: $TARGET_VCF_DIR"
    exit 1
fi


mkdir -p "$OUT_DIR"

CHUNK_DIR="$TARGET_VCF_DIR/chunks"
BCF_DIR="$TARGET_VCF_DIR/bcf"
XCF_DIR="$TARGET_VCF_DIR/xcf"
LOG_DIR="$OUT_DIR/logs"

for dir in "$BCF_DIR" "$CHUNK_DIR" "$XCF_DIR"
do
    if [ ! -d "$dir" ]; then
        echo "ERROR: Required directory does not exist: $dir"
        exit 1
    fi
done

if [ ! -d "$MAP_DIR" ]; then
    echo "ERROR: Map file does not exist: $MAP_DIR"
    exit 1
fi

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
    filename=${filename%.bcf}
    filename=${filename%.cleaned.target.bcf}
    filename=${filename%.target}
    filename=${filename%_target}
    filename=${filename%_target_onlyvariants}
    filename=${filename%_target_onlyvariants_phased}
    filename=${filename%_cleaned}

    echo "$filename"
}

index_vcf_if_needed() {
    vcf="$1"

    if [ ! -f "${vcf}.csi" ] && [ ! -f "${vcf}.tbi" ]; then
        bcftools index "$vcf"
    fi
}

index_bcf_if_needed() {
    bcf="$1"

    if [ ! -f "${bcf}.csi" ]; then
        bcftools index "$bcf"
    fi
}

error=0
no_error=0


for target_bcf in "$BCF_DIR"/*.cleaned.target.bcf
do
    [ -e "$target_bcf" ] || continue

    sample=$(get_sample_name "$target_bcf")
    sample="$sample.target.bcf"

    sample_chunk_dir="$CHUNK_DIR/$sample"
    sample_xcf_dir="$XCF_DIR/$sample"
    sample_impute_dir="$OUT_DIR/$sample"

    mkdir -p "$sample_impute_dir"

    log_file="$LOG_DIR/${sample}.impute5.log"
    : > "$log_file"

    log_msg "$log_file" "Starting per-chunk impute5 for sample: $sample"
    log_msg "$log_file" "Original target BCF: $target_bcf"
    log_msg "$log_file" "Chunk directory: $sample_chunk_dir"
    log_msg "$log_file" "Sample XCF directory: $sample_xcf_dir"
    log_msg "$log_file" "Impute output directory: $sample_impute_dir"

    if [ ! -d "$sample_chunk_dir" ]; then
        log_msg "$log_file" "Skipping sample because chunk directory does not exist: $sample_chunk_dir"
        continue
    fi

    if [ ! -d "$sample_xcf_dir" ]; then
        log_msg "$log_file" "Skipping sample because XCF directory does not exist: $sample_xcf_dir"
        continue
    fi

    for coordinates_file in "$sample_chunk_dir"/*.coordinates.txt
    do
        [ -e "$coordinates_file" ] || continue

        chr_from_file=$(basename "$coordinates_file")
        chr_from_file=${chr_from_file%.coordinates.txt}
        chr_from_file=${chr_from_file#${sample}.}

        log_msg "$log_file" "Using coordinates file: $coordinates_file"
        log_msg "$log_file" "Chromosome from coordinates filename: $chr_from_file"

        while read -r chunk_id chunk_chr buffered_region impute_region rest
        do
            [ -z "$chunk_id" ] && continue
            [[ "$chunk_id" =~ ^# ]] && continue

            if [ -z "$chunk_chr" ] || [ -z "$buffered_region" ] || [ -z "$impute_region" ]; then
                log_msg "$log_file" "Skipping malformed coordinates line: chunk_id=$chunk_id chunk_chr=$chunk_chr buffered_region=$buffered_region impute_region=$impute_region"
                continue
            fi

            chr="$chunk_chr"
            map_file="$MAP_DIR/${chr}_cleaned.map"

            if [ ! -f "$map_file" ]; then
                log_msg "$log_file" "Skipping chunk because map file does not exist: $map_file"
                continue
            fi

            if [ "$chr_from_file" != "$chr" ]; then
                log_msg "$log_file" "WARNING: chromosome in filename does not match chromosome in coordinates row"
                log_msg "$log_file" "Chromosome from file: $chr_from_file"
                log_msg "$log_file" "Chromosome from row: $chr"
            fi

            chr_ref_xcf="$XCF_DIR/reference_panel.${chr}.xcf.bcf"
            chr_target_xcf="$sample_xcf_dir/${sample}.${chr}.target.xcf.bcf"

            imputed_bcf="$sample_impute_dir/${sample}.${chr}.chunk${chunk_id}.imputed.bcf"
            imputed_log="$sample_impute_dir/${sample}.${chr}.chunk${chunk_id}.impute5.log"

            log_msg "$log_file" "Preparing impute5 chunk"
            log_msg "$log_file" "Sample: $sample"
            log_msg "$log_file" "Chromosome: $chr"
            log_msg "$log_file" "Chunk ID: $chunk_id"
            log_msg "$log_file" "Reference XCF: $chr_ref_xcf"
            log_msg "$log_file" "Target XCF: $chr_target_xcf"
            log_msg "$log_file" "Buffered region: $buffered_region"
            log_msg "$log_file" "Impute region: $impute_region"
            log_msg "$log_file" "Output BCF: $imputed_bcf"
            log_msg "$log_file" "Output log: $imputed_log"
            log_msg "$log_file" "Map: $map_file"

            if [ ! -f "$chr_ref_xcf" ]; then
                log_msg "$log_file" "Skipping chunk because reference XCF does not exist: $chr_ref_xcf"
                continue
            fi

            if [ ! -f "$chr_target_xcf" ]; then
                log_msg "$log_file" "Skipping chunk because target XCF does not exist: $chr_target_xcf"
                continue
            fi

            index_bcf_if_needed "$chr_ref_xcf"
            index_bcf_if_needed "$chr_target_xcf"

            if impute5 \
                --h "$chr_ref_xcf" \
                --g "$chr_target_xcf" \
                --r "$impute_region" \
                --buffer-region "$buffered_region" \
                --o "$imputed_bcf" \
                --m "$map_file" \
                --threads "$THREADS" \
                --l "$imputed_log" \
                >> "$log_file" 2>&1
            then
                log_msg "$log_file" "impute5 finished successfully for $sample $chr chunk $chunk_id"
                ((no_error++))
            else
                log_msg "$log_file" "ERROR: impute5 failed for  $sample $chr chunk $chunk_id"
                log_msg "$log_file" "See impute5 log: $imputed_log"
                if grep -q "Wrong order in your genetic map file" "$imputed_log"; then
                    ((error++))
                fi
                continue
            fi

            log_msg "$log_file" "Finished impute5 for $sample $chr chunk $chunk_id"

        done < "$coordinates_file"
    done
        
    log_msg "$log_file" "Finished per-chunk impute5 for sample: $sample"
    echo "Number of errors related to wrong order $error"
    echo "No errors: $no_error"
done