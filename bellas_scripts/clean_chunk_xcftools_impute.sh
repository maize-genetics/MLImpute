#!/bin/bash
set -uo pipefail

# REMINDER: make sure to retrieve the path to impute functions

export PATH=/programs/impute5_v1.2.0:$PATH

#test dir
#"/workdir/irk9/data/phg-maize/test/target"
#"/workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz"
#"/workdir/irk9/data/phg-maize/test/impute"

#TARGET_VCF_DIR="/workdir/irk9/data/phg-maize/target_vcf/0.01x"
#REF_PANEL_VCF="/workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz"

#OUT_DIR="/workdir/irk9/data/phg-maize/target_vcf/0.01x/impute"
#MAP_DIR="/path/to/genetic_maps"

# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-maize/target_vcf/0.01x --ref-panel-vcf /workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz --out-dir /workdir/irk9/data/phg-maize/target_vcf/0.01x/impute
# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-maize/target_vcf/0.1x --ref-panel-vcf /workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz --out-dir /workdir/irk9/data/phg-maize/target_vcf/0.1x/impute
# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-maize/target_vcf/2x --ref-panel-vcf /workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz --out-dir /workdir/irk9/data/phg-maize/target_vcf/2x/impute

# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-cassava/target_vcf/0.01x --ref-panel-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome.vcf.gz --out-dir /workdir/irk9/data/phg-cassava/target_vcf/0.01x/impute
# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-cassava/target_vcf/0.1x --ref-panel-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome.vcf.gz --out-dir /workdir/irk9/data/phg-cassava/target_vcf/0.1x/impute


# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-maize/test2/2x/target --ref-panel-vcf /workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz --out-dir /workdir/irk9/data/phg-maize/test2/2x/impute


# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-cassava/test2/2x/target --ref-panel-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_cleaned.vcf.gz --out-dir /workdir/irk9/data/phg-cassava/test2/2x/impute


# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-cassava/test2/5.07x/target --ref-panel-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_cleaned.vcf.gz  --out-dir /workdir/irk9/data/phg-cassava/test2/5.07x/impute

# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-maize/test2/5.07x/target --ref-panel-vcf /workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz --out-dir /workdir/irk9/data/phg-maize/test2/5.07x/impute


# cassava_test2 redo

# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-cassava/test2_redo/0.01x/target --ref-panel-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_cleaned.vcf.gz --out-dir /workdir/irk9/data/phg-cassava/test2_redo/0.01x/impute
# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-cassava/test2_redo/0.1x/target --ref-panel-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_cleaned.vcf.gz --out-dir /workdir/irk9/data/phg-cassava/test2_redo/0.1x/impute
# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-cassava/test2_redo/1x/target --ref-panel-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_cleaned.vcf.gz --out-dir /workdir/irk9/data/phg-cassava/test2_redo/1x/impute
# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-cassava/test2_redo/2x/target --ref-panel-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_cleaned.vcf.gz --out-dir /workdir/irk9/data/phg-cassava/test2_redo/2x/impute
# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-cassava/test2_redo/5.07x/target --ref-panel-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_cleaned.vcf.gz --out-dir /workdir/irk9/data/phg-cassava/test2_redo/5.07x/impute
# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-cassava/test2_redo/10x/target --ref-panel-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_cleaned.vcf.gz --out-dir /workdir/irk9/data/phg-cassava/test2_redo/10x/impute
# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-cassava/test2_redo/26.35x/target --ref-panel-vcf /workdir/irk9/data/phg-cassava/truth-vcfs/cassava_pangenome_diploid_cleaned.vcf.gz --out-dir /workdir/irk9/data/phg-cassava/test2_redo/26.35x/impute

# test 3
# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-maize/test3/2x/target --ref-panel-vcf /workdir/irk9/data/phg-maize/test3/maize_pangenome_snps_cleaned_phased_AN_AC_nomissing.vcf.gz --out-dir /workdir/irk9/data/phg-maize/test3/2x/impute

# ./clean_chunk_xcftools_impute.sh --step all --target-vcf-dir /workdir/irk9/data/phg-cassava/test3/0.1x/target --ref-panel-vcf /workdir/irk9/data/phg-cassava/test3/ref_panel/cassava_sim_merged_biallelic_diploid_header_AN_AC_nomissing_rename_chr.vcf.gz --out-dir /workdir/irk9/data/phg-cassava/test3/0.1x/impute

# ============================================================
# Default settings
# ============================================================

STEP="all"
TARGET_VCF_DIR=""
REF_PANEL_VCF="/workdir/irk9/data/phg-maize/test/ref_panel/maize_pangenome_snps_cleaned.vcf.gz"
OUT_DIR=""
THREADS=30
RENAME_CHR="/workdir/irk9/data/phg-cassava/test2/5.07x/rename_chr.txt"

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
    echo "  --step STEP               Step to run: clean, validate_cleaning, convert_bcf, make_xcf, make_chunks, impute, all"
    echo "                            Default: all"
    echo "  --map-dir DIR             Directory containing genetic maps, if needed"
    echo "  --threads INT             Number of threads"
    echo "                            Default: 4"
    echo "  -h, --help                Show this help message"
    echo
}

# ============================================================
# Parse command-line arguments
# ============================================================

while [[ $# -gt 0 ]]
do
    case "$1" in
        --step)
            STEP="$2"
            shift 2
            ;;

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

CLEAN_DIR="$OUT_DIR/cleaned_vcfs"
VALIDATION_DIR="$OUT_DIR/validation"
BCF_DIR="$OUT_DIR/bcf"
CHUNK_DIR="$OUT_DIR/chunks"
IMPUTE_DIR="$OUT_DIR/imputed"
LOG_DIR="$OUT_DIR/logs"

mkdir -p \
    "$CLEAN_DIR" \
    "$VALIDATION_DIR" \
    "$BCF_DIR" \
    "$CHUNK_DIR" \
    "$IMPUTE_DIR" \
    "$LOG_DIR"


#log_file_path="$LOG_DIR/arguments.log"
#: > "$log_file_path"
#log_msg "$log_file_path" "Target: $TARGET_VCF_DIR \n Reference_panel: $REF_PANEL_VCF \n Output: $OUT_DIR"



# ============================================================
# Choose which section to run
# ============================================================
# Options:
#   clean
#   validate_cleaning
#   convert_bcf
#   make_chunks
#   impute
#   all

#STEP="${1:-all}"

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

# ============================================================
# Section 1: Clean target VCFs
# ============================================================
# Cleaning does:
#   1. fix ploidy
#   2. replace / with |
#   3. remove genotypes with missing alleles
#
# Output:
#   sample_cleaned.vcf.gz

clean_vcfs() {
    for input_vcf in "$TARGET_VCF_DIR"/*.vcf.gz
    do
        [ -e "$input_vcf" ] || continue

        sample=$(get_sample_name "$input_vcf")
        log_file="$LOG_DIR/${sample}.cleaning.log"

        rename_chr="$CLEAN_DIR/${sample}.rename_chr.vcf.gz"
        fixed_vcf="$CLEAN_DIR/${sample}.fixploidy.vcf.gz"
        phased_vcf="$CLEAN_DIR/${sample}.phased.vcf.gz"
        cleaned_vcf="$CLEAN_DIR/${sample}.cleaned.vcf.gz"


        : > "$log_file"

        log_msg "$log_file" "Starting cleaning for sample: $sample"
        log_msg "$log_file" "Input VCF: $input_vcf"
        log_msg "$log_file" "Fixploidy output: $fixed_vcf"
        log_msg "$log_file" "Phased slash-to-pipe output: $phased_vcf"
        log_msg "$log_file" "Final cleaned output: $cleaned_vcf"


        #For cassava files only
        bcftools annotate --rename-chrs "$RENAME_CHR" -Oz -o "$rename_chr" "$input_vcf"

        
        bcftools index -f "$rename_chr" >> "$log_file" 2>&1


        # ----------------------------------------------------
        # 1A. Make haploid calls diploid
        # ----------------------------------------------------
        #
        # You may need to adjust plugin options depending on your data.
        # The important part is that this section is isolated so you can validate it.
        #
        # Example structure:
        # bcftools +fixploidy input.vcf.gz -Oz -o output.vcf.gz -- [Plugin Options]

        log_msg "$log_file" "Running fixploidy"

        bcftools +fixploidy "${rename_chr}" \
            -Oz \
            -o "$fixed_vcf" \
            >> "$log_file" 2>&1

        bcftools index "$fixed_vcf" >> "$log_file" 2>&1

        # ----------------------------------------------------
        # 1B. Replace / with | in genotype field
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 1C. Remove sites with missing genotypes
        # ----------------------------------------------------
        #
        # This removes records where any genotype is missing.
        # For a single-sample target VCF, this removes records like:
        #   .|.
        #   .|0
        #   0|.
        #   ./.
        #   ./0
        #   0/.

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
}

# ============================================================
# Section 2: Validate cleaned VCFs
# ============================================================

validate_cleaning() {
    for cleaned_vcf in "$CLEAN_DIR"/*.cleaned.vcf.gz
    do
        [ -e "$cleaned_vcf" ] || continue

        sample=$(get_sample_name "$cleaned_vcf")
        log_file="$LOG_DIR/${sample}.validate_cleaning.log"
        report_file="$VALIDATION_DIR/${sample}.cleaning_validation.txt"

        : > "$log_file"
        : > "$report_file"

        log_msg "$log_file" "Validating cleaned VCF for sample: $sample"
        log_msg "$log_file" "Cleaned VCF: $cleaned_vcf"
        log_msg "$log_file" "Report file: $report_file"

        {
            echo "Sample: $sample"
            echo "Cleaned VCF: $cleaned_vcf"
            echo

            echo "Number of records:"
            bcftools view -H "$cleaned_vcf" | wc -l
            echo

            echo "First 10 variant records:"
            bcftools view -H "$cleaned_vcf" | head
            echo

            echo "Checking for unphased slash genotypes:"
            bcftools query -f '[%GT\n]' "$cleaned_vcf" | grep "/" | head || true
            echo

            echo "Checking for missing genotypes containing dot:"
            bcftools query -f '[%GT\n]' "$cleaned_vcf" | grep "\." | head || true
            echo

            echo "Genotype counts:"
            bcftools query -f '[%GT\n]' "$cleaned_vcf" | sort | uniq -c | sort -nr | head -20
            echo

        } >> "$report_file" 2>> "$log_file"

        log_msg "$log_file" "Finished validation for sample: $sample"
    done
}

# ============================================================
# Section 3: Convert cleaned target VCFs and reference panel to BCF
# ============================================================

convert_to_bcf() {
    ref_log="$LOG_DIR/reference_panel.convert_bcf.log"
    ref_bcf="$BCF_DIR/reference_panel.bcf"

    : > "$ref_log"

    log_msg "$ref_log" "Converting reference panel to BCF"
    log_msg "$ref_log" "Reference panel VCF: $REF_PANEL_VCF"
    log_msg "$ref_log" "Reference panel BCF: $ref_bcf"

    bcftools view "$REF_PANEL_VCF" \
        -Ob \
        -o "$ref_bcf" \
        >> "$ref_log" 2>&1

    bcftools index "$ref_bcf" >> "$ref_log" 2>&1

    for cleaned_vcf in "$CLEAN_DIR"/*.cleaned.vcf.gz
    do
        [ -e "$cleaned_vcf" ] || continue

        sample=$(get_sample_name "$cleaned_vcf")
        target_bcf="$BCF_DIR/${sample}.target.bcf"
        log_file="$LOG_DIR/${sample}.convert_bcf.log"

        : > "$log_file"

        log_msg "$log_file" "Converting cleaned target VCF to BCF"
        log_msg "$log_file" "Input cleaned VCF: $cleaned_vcf"
        log_msg "$log_file" "Output target BCF: $target_bcf"

        bcftools view "$cleaned_vcf" \
            -Ob \
            -o "$target_bcf" \
            >> "$log_file" 2>&1

        bcftools index "$target_bcf" >> "$log_file" 2>&1

        log_msg "$log_file" "Finished BCF conversion for sample: $sample"
    done
}

# ============================================================
# Section 4: Make chunks with imp5Chunker
# ============================================================
#
# This section:
#   1. Finds chromosomes in target BCF
#   2. Checks that the reference panel has the same chromosome
#   3. Gets overlapping start and end positions
#   4. Runs imp5Chunker
#
# Region format:
#   chr1:13000-5000000
#   or
#   1:13000-5000000

make_chunks() {
    ref_bcf="$BCF_DIR/reference_panel.bcf"
    index_bcf_if_needed "$ref_bcf"

    for target_bcf in "$BCF_DIR"/*.target.bcf
    do
        [ -e "$target_bcf" ] || continue

        sample=$(get_sample_name "$target_bcf")
        log_file="$LOG_DIR/${sample}.make_chunks.log"

        sample_chunk_dir="$CHUNK_DIR/$sample"
        mkdir -p "$sample_chunk_dir"

        : > "$log_file"

        log_msg "$log_file" "Making chunks for sample: $sample"
        log_msg "$log_file" "Reference BCF: $ref_bcf"
        log_msg "$log_file" "Target BCF: $target_bcf"
        log_msg "$log_file" "Chunk output directory: $sample_chunk_dir"

        chromosomes=$(bcftools query -f '%CHROM\n' "$target_bcf" | sort -u)

        for chr in $chromosomes
        do
            log_msg "$log_file" "Checking chromosome: $chr"

            ref_count=$(bcftools view -H -r "$chr" "$ref_bcf" | awk 'END {print NR}')
            echo "Ref count: $ref_count"

            target_count=$(bcftools view -H -r "$chr" "$target_bcf" | awk 'END {print NR}')
            echo "Target count: $target_count"

            if [ "$ref_count" -eq 0 ]; then
                log_msg "$log_file" "Skipping $chr because it is not present in reference panel"
                continue
            fi

            if [ "$target_count" -eq 0 ]; then
                log_msg "$log_file" "Skipping $chr because it is not present in target BCF"
                continue
            fi

            ref_start=$(bcftools view -H -r "$chr" "$ref_bcf" | awk 'NR==1 {print $2}')
            ref_end=$(bcftools view -H -r "$chr" "$ref_bcf" | awk 'END {print $2}')
            echo "Ref_interval: $ref_start - $ref_end"

            target_start=$(bcftools view -H -r "$chr" "$target_bcf" | awk 'NR==1 {print $2}')
            target_end=$(bcftools view -H -r "$chr" "$target_bcf" | awk 'END {print $2}')
            echo "Target_interval: $target_start - $target_end"

            start=$(awk -v a="$ref_start" -v b="$target_start" 'BEGIN { if (a > b) print a; else print b }')
            end=$(awk -v a="$ref_end" -v b="$target_end" 'BEGIN { if (a < b) print a; else print b }')

            echo "Shared_Interval: $start - $end"

            if [ "$start" -ge "$end" ]; then
                log_msg "$log_file" "Skipping $chr because there is no overlapping interval"
                log_msg "$log_file" "Reference interval: $chr:$ref_start-$ref_end"
                log_msg "$log_file" "Target interval: $chr:$target_start-$target_end"
                continue
            fi

            region="${chr}:${start}-${end}"
            echo "Region: $region"
            coordinates_file="$sample_chunk_dir/${sample}.${chr}.coordinates.txt"
            echo "Coordinates file: $coordinates_file"

            log_msg "$log_file" "Reference interval: $chr:$ref_start-$ref_end"
            log_msg "$log_file" "Target interval: $chr:$target_start-$target_end"
            log_msg "$log_file" "Using overlap region: $region"
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
}

# ============================================================
# Section 4: Convert BCF to XCF.BCF with xcftools
# ============================================================
#
# Reference:
#   xcftools view -i reference.bcf -o reference_xcf.bcf -O sh -r chr:start-end -T8 -m 0.03125
#
# Target:
#   xcftools view -i target.bcf -o target_xcf.bcf -O bh -r chr:start-end -T8 -m 0.03125
#
# Notes:
#   - Reference uses: -O sh
#   - Target uses:    -O bh
#   - Region is based on each file's own start and end position.
#   - This does NOT use the overlap between reference and target.
# ============================================================

XCF_DIR="$OUT_DIR/xcf"
mkdir -p "$XCF_DIR"

make_xcf() {
    ref_bcf="$BCF_DIR/reference_panel.bcf"
    ref_xcf="$XCF_DIR/reference_panel.xcf.bcf"
    ref_log="$LOG_DIR/reference_panel.make_xcf.log"

    : > "$ref_log"

    log_msg "$ref_log" "Starting XCF conversion for reference panel"
    log_msg "$ref_log" "Input reference BCF: $ref_bcf"
    log_msg "$ref_log" "Output reference XCF BCF: $ref_xcf"
    log_msg "$ref_log" "Reference output mode: -O sh"
    log_msg "$ref_log" "Threads: $THREADS"
    log_msg "$ref_log" "Minor allele frequency threshold: 0.03125"

    index_bcf_if_needed "$ref_bcf"

    # --------------------------------------------------------
    # Convert reference panel by chromosome
    # --------------------------------------------------------

    ref_chromosomes=$(bcftools query -f '%CHROM\n' "$ref_bcf" | sort -u)

    for chr in $ref_chromosomes
    do

        if [[ "$chr" != chr* ]]; then
            log_msg "$ref_log" "Skipping $chr because it does not start with chr"
            continue
        fi

        chr_ref_xcf="$XCF_DIR/reference_panel.${chr}.xcf.bcf"

        ref_start=$(bcftools view -H -r "$chr" "$ref_bcf" | awk 'NR==1 {print $2}')
        ref_end=$(bcftools view -H -r "$chr" "$ref_bcf" | awk 'END {print $2}')

        if [ -z "$ref_start" ] || [ -z "$ref_end" ]; then
            log_msg "$ref_log" "Skipping $chr because no start/end position was found"
            continue
        fi

        region="${chr}:${ref_start}-${ref_end}"

        log_msg "$ref_log" "Converting reference chromosome: $chr"
        log_msg "$ref_log" "Reference start: $ref_start"
        log_msg "$ref_log" "Reference end: $ref_end"
        log_msg "$ref_log" "Reference region: $region"
        log_msg "$ref_log" "Output file: $chr_ref_xcf"

        xcftools view \
            -i "$ref_bcf" \
            -o "$chr_ref_xcf" \
            -O sh \
            -r "$region" \
            -T "$THREADS" \
            -m 0.03125 \
            >> "$ref_log" 2>&1

        log_msg "$ref_log" "Finished reference XCF conversion for $chr"
    done

    # --------------------------------------------------------
    # Convert each target BCF by chromosome
    # --------------------------------------------------------

    for target_bcf in "$BCF_DIR"/*.target.bcf
    do
        [ -e "$target_bcf" ] || continue

        sample=$(get_sample_name "$target_bcf")
        target_log="$LOG_DIR/${sample}.make_xcf.log"
        sample_xcf_dir="$XCF_DIR/$sample"

        mkdir -p "$sample_xcf_dir"
        : > "$target_log"

        log_msg "$target_log" "Starting XCF conversion for target sample: $sample"
        log_msg "$target_log" "Input target BCF: $target_bcf"
        log_msg "$target_log" "Target output mode: -O bh"
        log_msg "$target_log" "Threads: $THREADS"
        log_msg "$target_log" "Minor allele frequency threshold: 0.03125"

        index_bcf_if_needed "$target_bcf"

        target_chromosomes=$(bcftools query -f '%CHROM\n' "$target_bcf" | sort -u)

        for chr in $target_chromosomes
        do
            if [[ "$chr" != chr* ]]; then
                log_msg "$target_log" "Skipping $chr because it does not start with chr"
                continue
            fi

            chr_target_xcf="$sample_xcf_dir/${sample}.${chr}.target.xcf.bcf"

            target_start=$(bcftools view -H -r "$chr" "$target_bcf" | awk 'NR==1 {print $2}')
            target_end=$(bcftools view -H -r "$chr" "$target_bcf" | awk 'END {print $2}')

            if [ -z "$target_start" ] || [ -z "$target_end" ]; then
                log_msg "$target_log" "Skipping $chr because no start/end position was found"
                continue
            fi

            region="${chr}:${target_start}-${target_end}"

            log_msg "$target_log" "Converting target chromosome: $chr"
            log_msg "$target_log" "Target start: $target_start"
            log_msg "$target_log" "Target end: $target_end"
            log_msg "$target_log" "Target region: $region"
            log_msg "$target_log" "Output file: $chr_target_xcf"

            xcftools view \
                -i "$target_bcf" \
                -o "$chr_target_xcf" \
                -O bh \
                -r "$region" \
                -T "$THREADS" \
                -m 0.03125 \
                >> "$target_log" 2>&1

            log_msg "$target_log" "Finished target XCF conversion for $sample $chr"
        done

        log_msg "$target_log" "Finished all target XCF conversion for sample: $sample"
    done
}

# ============================================================
# Section 5: Run impute5
# ============================================================
#
# This assumes:
#   - You don't have a genetic map. If maps poriton would be implemented, assume its one map per species.
#   - Chunk files were generated by imp5Chunker
#

run_impute5() {

    for target_bcf in "$BCF_DIR"/*.target.bcf
    do
        [ -e "$target_bcf" ] || continue

        sample=$(get_sample_name "$target_bcf")

        sample_chunk_dir="$CHUNK_DIR/$sample"
        sample_xcf_dir="$XCF_DIR/$sample"
        sample_impute_dir="$IMPUTE_DIR/$sample"

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
                    --l "$imputed_log" \
                    >> "$log_file" 2>&1
                then
                    log_msg "$log_file" "impute5 finished successfully for $sample $chr chunk $chunk_id"
                else
                    log_msg "$log_file" "ERROR: impute5 failed for $sample $chr chunk $chunk_id"
                    log_msg "$log_file" "See impute5 log: $imputed_log"
                    continue
                fi
                

                #echo "indexing"

                #bcftools index "$imputed_bcf" >> "$log_file" 2>&1

                log_msg "$log_file" "Finished impute5 for $sample $chr chunk $chunk_id"

            done < "$coordinates_file"
        done
        
        log_msg "$log_file" "Finished per-chunk impute5 for sample: $sample"
    done
}

# ============================================================
# Run selected step
# ============================================================

case "$STEP" in
    clean)
        clean_vcfs
        ;;

    validate_cleaning)
        validate_cleaning
        ;;

    convert_bcf)
        convert_to_bcf
        ;;

    make_chunks)
        make_chunks
        ;;
    
    make_xcf)
        make_xcf
        ;;

    impute)
        run_impute5
        ;;

    all)
        clean_vcfs
        validate_cleaning
        convert_to_bcf
        make_chunks
        make_xcf
        run_impute5
        ;;

    *)
        echo "Unknown step: $STEP"
        echo "Valid options: clean, validate_cleaning, convert_bcf, make_chunks, impute, all"
        exit 1
        ;;
esac
