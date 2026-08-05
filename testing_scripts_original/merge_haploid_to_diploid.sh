#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 -i <input.vcf.gz> -s <new_sample_name> -o <output.vcf.gz>"
    exit 1
}

input=""
sample_name=""
output=""

while getopts "i:s:o:" opt; do
    case $opt in
        i) input="$OPTARG" ;;
        s) sample_name="$OPTARG" ;;
        o) output="$OPTARG" ;;
        *) usage ;;
    esac
done

[[ -z "$input" || -z "$sample_name" || -z "$output" ]] && usage

bcftools view "$input" | awk -v new_name="$sample_name" '
  BEGIN { OFS="\t" }
  /^##/ { print; next }
  /^#CHROM/ {
      print $0 "\t" new_name
      next
  }
  {
      new_col = $10 "/" $11
      print $0 "\t" new_col
  }' | bgzip > "$output"
