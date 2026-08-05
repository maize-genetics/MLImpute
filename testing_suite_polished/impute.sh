#!/bin/bash
set -uo pipefail

export PATH=/programs/impute5_v1.2.0:$PATH

# ============================================================
# Default settings
# ============================================================

CLEAN_DIR=""        # directory with *.cleaned.vcf.gz (output of clean.sh)
REF_PANEL_VCF=""
OUT_DIR=""
SPECIES=""          # maize | cassava
MAP_DIR=""          # optional; when set, passes --m to impute5
THREADS=5

# ============================================================
# Help message
# ============================================================

usage() {
    echo
    echo "Usage:"
    echo "  $0 --clean-dir DIR --ref-panel-vcf FILE --out-dir DIR --species SPECIES [options]"
    echo
    echo "Required arguments:"
    echo "  --clean-dir DIR           Directory with *.cleaned.vcf.gz (output of clean.sh)"
    echo "  --ref-panel-vcf FILE      Reference panel VCF file"
    echo "  --out-dir DIR             Output directory"
    echo "  --species SPECIES         Species: maize or cassava"
    echo
    echo "Optional arguments:"
    echo "  --map-dir DIR             Directory with genetic maps; when provided, passes --m to impute5"
    echo "  --threads INT             Number of threads (default: 5)"
    echo "  -h, --help                Show this help message"
    echo
}

# ./impute.sh --clean-dir /workdir/irk9/data/phg-maize/bellas_scripts/testing_suite_bella/testing_suite_test/target_cleaned --ref-panel-vcf /workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz --out-dir /workdir/irk9/data/phg-maize/bellas_scripts/testing_suite_bella/testing_suite_test/impute --species maize 

# ============================================================
# Parse command-line arguments
# ============================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean-dir)      CLEAN_DIR="$2";     shift 2 ;;
        --ref-panel-vcf)  REF_PANEL_VCF="$2"; shift 2 ;;
        --out-dir)        OUT_DIR="$2";       shift 2 ;;
        --species)        SPECIES="$2";       shift 2 ;;
        --map-dir)        MAP_DIR="$2";       shift 2 ;;
        --threads)        THREADS="$2";       shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

# ============================================================
# Validate required arguments
# ============================================================

if [ -z "$CLEAN_DIR" ];     then echo "ERROR: Missing required argument: --clean-dir";     usage; exit 1; fi
if [ -z "$REF_PANEL_VCF" ]; then echo "ERROR: Missing required argument: --ref-panel-vcf"; usage; exit 1; fi
if [ -z "$OUT_DIR" ];        then echo "ERROR: Missing required argument: --out-dir";        usage; exit 1; fi
if [ -z "$SPECIES" ];        then echo "ERROR: Missing required argument: --species";         usage; exit 1; fi

if [ "$SPECIES" != "maize" ] && [ "$SPECIES" != "cassava" ]; then
    echo "ERROR: --species must be maize or cassava, got: $SPECIES"
    exit 1
fi

# ============================================================
# Validate paths
# ============================================================

if [ ! -d "$CLEAN_DIR" ];     then echo "ERROR: Clean directory does not exist: $CLEAN_DIR";          exit 1; fi
if [ ! -f "$REF_PANEL_VCF" ]; then echo "ERROR: Reference panel VCF does not exist: $REF_PANEL_VCF"; exit 1; fi
if [ -n "$MAP_DIR" ] && [ ! -d "$MAP_DIR" ]; then echo "ERROR: Map directory does not exist: $MAP_DIR"; exit 1; fi

# ============================================================
# Species-specific settings
# ============================================================

# maize chromosomes: chr1, chr2, ...
# cassava chromosomes: Chr01, Chr02, ...
if [ "$SPECIES" = "maize" ]; then
    CHR_PREFIX="chr"
else
    CHR_PREFIX="Chr"
fi

# ============================================================
# Set up output directories
# ============================================================

mkdir -p "$OUT_DIR"

BCF_DIR="$OUT_DIR/bcf"
CHUNK_DIR="$OUT_DIR/chunks"
XCF_DIR="$OUT_DIR/xcf"
IMPUTE_DIR="$OUT_DIR/imputed"
LOG_DIR="$OUT_DIR/logs"

mkdir -p "$BCF_DIR" "$CHUNK_DIR" "$XCF_DIR" "$IMPUTE_DIR" "$LOG_DIR"

# ============================================================
# Helper functions
# ============================================================

log_msg() {
    local log_file="$1"
    local msg="$2"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$log_file"
}

get_sample_name() {
    local filename
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

index_bcf_if_needed() {
    local bcf="$1"
    if [ ! -f "${bcf}.csi" ]; then
        bcftools index "$bcf"
    fi
}

# ============================================================
# Step 1: Convert cleaned target VCFs and ref panel to BCF
# ============================================================

ref_log="$LOG_DIR/reference_panel.convert_bcf.log"
ref_bcf="$BCF_DIR/reference_panel.bcf"
: > "$ref_log"

log_msg "$ref_log" "Converting reference panel to BCF"
log_msg "$ref_log" "Input:  $REF_PANEL_VCF"
log_msg "$ref_log" "Output: $ref_bcf"

if [ ! -f "$ref_bcf" ]; then
    bcftools view "$REF_PANEL_VCF" -Ob -o "$ref_bcf" >> "$ref_log" 2>&1
    bcftools index "$ref_bcf" >> "$ref_log" 2>&1
else
    log_msg "$ref_log" "Skipping: reference panel BCF already exists"
    index_bcf_if_needed "$ref_bcf"
fi

for cleaned_vcf in "$CLEAN_DIR"/*.cleaned.vcf.gz; do
    [ -e "$cleaned_vcf" ] || continue

    sample=$(get_sample_name "$cleaned_vcf")
    target_bcf="$BCF_DIR/${sample}.target.bcf"
    log_file="$LOG_DIR/${sample}.convert_bcf.log"
    : > "$log_file"

    log_msg "$log_file" "Converting cleaned VCF to BCF: $sample"
    log_msg "$log_file" "Input:  $cleaned_vcf"
    log_msg "$log_file" "Output: $target_bcf"

    if [ ! -f "$target_bcf" ]; then
        bcftools view "$cleaned_vcf" -Ob -o "$target_bcf" >> "$log_file" 2>&1
        bcftools index "$target_bcf" >> "$log_file" 2>&1
    else
        log_msg "$log_file" "Skipping: target BCF already exists"
        index_bcf_if_needed "$target_bcf"
    fi

    log_msg "$log_file" "Finished BCF conversion for sample: $sample"
done

# ============================================================
# Step 2: Make chunks with imp5Chunker
# ============================================================

index_bcf_if_needed "$ref_bcf"

for target_bcf in "$BCF_DIR"/*.target.bcf; do
    [ -e "$target_bcf" ] || continue

    sample=$(get_sample_name "$target_bcf")
    log_file="$LOG_DIR/${sample}.make_chunks.log"
    sample_chunk_dir="$CHUNK_DIR/$sample"
    mkdir -p "$sample_chunk_dir"
    : > "$log_file"

    log_msg "$log_file" "Making chunks for sample: $sample"

    chromosomes=$(bcftools query -f '%CHROM\n' "$target_bcf" | sort -u)

    for chr in $chromosomes; do
        [[ "$chr" != ${CHR_PREFIX}* ]] && continue

        coordinates_file="$sample_chunk_dir/${sample}.${chr}.coordinates.txt"

        if [ -f "$coordinates_file" ]; then
            log_msg "$log_file" "Skipping $chr: coordinates file already exists"
            continue
        fi

        ref_count=$(bcftools view -H -r "$chr" "$ref_bcf" | awk 'END {print NR}')
        target_count=$(bcftools view -H -r "$chr" "$target_bcf" | awk 'END {print NR}')

        if [ "$ref_count" -eq 0 ]; then
            log_msg "$log_file" "Skipping $chr: not present in reference panel"
            continue
        fi
        if [ "$target_count" -eq 0 ]; then
            log_msg "$log_file" "Skipping $chr: not present in target BCF"
            continue
        fi

        ref_start=$(bcftools view -H -r "$chr" "$ref_bcf" | awk 'NR==1 {print $2}')
        ref_end=$(bcftools view -H -r "$chr" "$ref_bcf" | awk 'END {print $2}')
        target_start=$(bcftools view -H -r "$chr" "$target_bcf" | awk 'NR==1 {print $2}')
        target_end=$(bcftools view -H -r "$chr" "$target_bcf" | awk 'END {print $2}')

        start=$(awk -v a="$ref_start" -v b="$target_start" 'BEGIN { print (a > b) ? a : b }')
        end=$(awk -v a="$ref_end" -v b="$target_end" 'BEGIN { print (a < b) ? a : b }')

        if [ "$start" -ge "$end" ]; then
            log_msg "$log_file" "Skipping $chr: no overlapping interval (ref $ref_start-$ref_end, target $target_start-$target_end)"
            continue
        fi

        region="${chr}:${start}-${end}"
        log_msg "$log_file" "Region: $region"
        log_msg "$log_file" "Coordinates file: $coordinates_file"

        imp5Chunker \
            --h "$ref_bcf" \
            --g "$target_bcf" \
            --r "$region" \
            --o "$coordinates_file" \
            >> "$log_file" 2>&1

        log_msg "$log_file" "Finished imp5Chunker for $sample $chr"
    done
done

# ============================================================
# Step 3: Convert BCF to XCF with xcftools
# ============================================================

ref_log="$LOG_DIR/reference_panel.make_xcf.log"
: > "$ref_log"

log_msg "$ref_log" "Starting XCF conversion for reference panel"
index_bcf_if_needed "$ref_bcf"

ref_chromosomes=$(bcftools query -f '%CHROM\n' "$ref_bcf" | sort -u)

for chr in $ref_chromosomes; do
    [[ "$chr" != ${CHR_PREFIX}* ]] && continue

    chr_ref_xcf="$XCF_DIR/reference_panel.${chr}.xcf.bcf"

    if [ -f "$chr_ref_xcf" ]; then
        log_msg "$ref_log" "Skipping $chr: reference XCF already exists"
        continue
    fi

    ref_start=$(bcftools view -H -r "$chr" "$ref_bcf" | awk 'NR==1 {print $2}')
    ref_end=$(bcftools view -H -r "$chr" "$ref_bcf" | awk 'END {print $2}')

    if [ -z "$ref_start" ] || [ -z "$ref_end" ]; then
        log_msg "$ref_log" "Skipping $chr: no start/end position found"
        continue
    fi

    region="${chr}:${ref_start}-${ref_end}"
    log_msg "$ref_log" "Converting reference $chr: $region → $chr_ref_xcf"

    xcftools view \
        -i "$ref_bcf" \
        -o "$chr_ref_xcf" \
        -O sh \
        -r "$region" \
        -T "$THREADS" \
        -m 0.03125 \
        >> "$ref_log" 2>&1

    log_msg "$ref_log" "Finished reference XCF for $chr"
done

for target_bcf in "$BCF_DIR"/*.target.bcf; do
    [ -e "$target_bcf" ] || continue

    sample=$(get_sample_name "$target_bcf")
    target_log="$LOG_DIR/${sample}.make_xcf.log"
    sample_xcf_dir="$XCF_DIR/$sample"
    mkdir -p "$sample_xcf_dir"
    : > "$target_log"

    log_msg "$target_log" "Starting XCF conversion for target: $sample"
    index_bcf_if_needed "$target_bcf"

    target_chromosomes=$(bcftools query -f '%CHROM\n' "$target_bcf" | sort -u)

    for chr in $target_chromosomes; do
        [[ "$chr" != ${CHR_PREFIX}* ]] && continue

        chr_target_xcf="$sample_xcf_dir/${sample}.${chr}.target.xcf.bcf"

        if [ -f "$chr_target_xcf" ]; then
            log_msg "$target_log" "Skipping $chr: target XCF already exists"
            continue
        fi

        target_start=$(bcftools view -H -r "$chr" "$target_bcf" | awk 'NR==1 {print $2}')
        target_end=$(bcftools view -H -r "$chr" "$target_bcf" | awk 'END {print $2}')

        if [ -z "$target_start" ] || [ -z "$target_end" ]; then
            log_msg "$target_log" "Skipping $chr: no start/end position found"
            continue
        fi

        region="${chr}:${target_start}-${target_end}"
        log_msg "$target_log" "Converting target $chr: $region → $chr_target_xcf"

        xcftools view \
            -i "$target_bcf" \
            -o "$chr_target_xcf" \
            -O bh \
            -r "$region" \
            -T "$THREADS" \
            -m 0.03125 \
            >> "$target_log" 2>&1

        log_msg "$target_log" "Finished target XCF for $sample $chr"
    done

    log_msg "$target_log" "Finished all target XCF conversion for sample: $sample"
done

# ============================================================
# Step 4: Run impute5
# ============================================================

error=0
no_error=0

for target_bcf in "$BCF_DIR"/*.target.bcf; do
    [ -e "$target_bcf" ] || continue

    sample=$(get_sample_name "$target_bcf")
    sample_chunk_dir="$CHUNK_DIR/$sample"
    sample_xcf_dir="$XCF_DIR/$sample"
    sample_impute_dir="$IMPUTE_DIR/$sample"
    mkdir -p "$sample_impute_dir"

    log_file="$LOG_DIR/${sample}.impute5.log"
    : > "$log_file"

    log_msg "$log_file" "Starting per-chunk impute5 for sample: $sample"

    if [ ! -d "$sample_chunk_dir" ]; then
        log_msg "$log_file" "Skipping: chunk directory does not exist: $sample_chunk_dir"
        continue
    fi

    if [ ! -d "$sample_xcf_dir" ]; then
        log_msg "$log_file" "Skipping: XCF directory does not exist: $sample_xcf_dir"
        continue
    fi

    for coordinates_file in "$sample_chunk_dir"/*.coordinates.txt; do
        [ -e "$coordinates_file" ] || continue

        chr_from_file=$(basename "$coordinates_file")
        chr_from_file=${chr_from_file%.coordinates.txt}
        chr_from_file=${chr_from_file#${sample}.}

        log_msg "$log_file" "Using coordinates file: $coordinates_file"

        while read -r chunk_id chunk_chr buffered_region impute_region rest; do
            [ -z "$chunk_id" ] && continue
            [[ "$chunk_id" =~ ^# ]] && continue

            if [ -z "$chunk_chr" ] || [ -z "$buffered_region" ] || [ -z "$impute_region" ]; then
                log_msg "$log_file" "Skipping malformed line: chunk_id=$chunk_id"
                continue
            fi

            chr="$chunk_chr"

            if [ "$chr_from_file" != "$chr" ]; then
                log_msg "$log_file" "WARNING: chr in filename ($chr_from_file) != chr in row ($chr)"
            fi

            chr_ref_xcf="$XCF_DIR/reference_panel.${chr}.xcf.bcf"
            chr_target_xcf="$sample_xcf_dir/${sample}.${chr}.target.xcf.bcf"
            imputed_bcf="$sample_impute_dir/${sample}.${chr}.chunk${chunk_id}.imputed.bcf"
            imputed_log="$sample_impute_dir/${sample}.${chr}.chunk${chunk_id}.impute5.log"

            if [ -f "$imputed_bcf" ]; then
                log_msg "$log_file" "Skipping: $chr chunk $chunk_id already imputed"
                continue
            fi

            if [ ! -f "$chr_ref_xcf" ]; then
                log_msg "$log_file" "Skipping: reference XCF does not exist: $chr_ref_xcf"
                continue
            fi

            if [ ! -f "$chr_target_xcf" ]; then
                log_msg "$log_file" "Skipping: target XCF does not exist: $chr_target_xcf"
                continue
            fi

            index_bcf_if_needed "$chr_ref_xcf"
            index_bcf_if_needed "$chr_target_xcf"

            log_msg "$log_file" "Imputing $sample $chr chunk $chunk_id"
            log_msg "$log_file" "  buffered=$buffered_region impute=$impute_region"

            impute5_cmd=(
                impute5
                --h "$chr_ref_xcf"
                --g "$chr_target_xcf"
                --r "$impute_region"
                --buffer-region "$buffered_region"
                --o "$imputed_bcf"
                --l "$imputed_log"
                --threads "$THREADS"
            )

            if [ -n "$MAP_DIR" ]; then
                map_file="$MAP_DIR/${chr}_cleaned.map"
                if [ ! -f "$map_file" ]; then
                    log_msg "$log_file" "Skipping: map file does not exist: $map_file"
                    continue
                fi
                impute5_cmd+=(--m "$map_file")
                log_msg "$log_file" "  map=$map_file"
            fi

            if "${impute5_cmd[@]}" >> "$log_file" 2>&1; then
                log_msg "$log_file" "impute5 succeeded: $sample $chr chunk $chunk_id"
                ((no_error++))
            else
                log_msg "$log_file" "ERROR: impute5 failed: $sample $chr chunk $chunk_id"
                log_msg "$log_file" "See impute5 log: $imputed_log"
                if grep -q "Wrong order in your genetic map file" "$imputed_log" 2>/dev/null; then
                    ((error++))
                fi
            fi

        done < "$coordinates_file"
    done

    log_msg "$log_file" "Finished per-chunk impute5 for sample: $sample"
done
