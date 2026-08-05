############
# DONT USE #
############

#!/usr/bin/env bash

base_dir="/workdir/irk9/data/phg-maize/test/impute/imputed/B97.cleaned.target.bcf"
output="${base_dir}/vcf_list.txt"

# Empty the output file if it already exists
#> "$output"

for file in "$base_dir"/*.imputed.bcf
do
    # Get just the filename, not the full path
    filename=$(basename "$file")

    # Match filenames containing: chr<number>.chunk<number>
    # Example:
    # B97.cleaned.target.bcf.chr1.chunk12.imputed.bcf
    regex='chr([0-9]+)\.chunk([0-9]+)'

    if [[ "$filename" =~ $regex ]]; then
        chr="${BASH_REMATCH[1]}"
        chunk="${BASH_REMATCH[2]}"

        # Print temporary sortable columns:
        # chr chunk full_file_path
        echo "$chr $chunk $file"
    else
        echo "Warning: could not parse filename: $filename" >&2
    fi
done \
| sort -k1,1n -k2,2n \
| cut -d' ' -f3- \
> "$output"

echo "Wrote sorted VCF/BCF list to: $output"
cat "$output"