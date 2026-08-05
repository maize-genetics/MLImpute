#!/bin/bash

# Folder containing FASTA files
FASTA_DIR="/workdir/irk9/data/phg-maize/gvcf-output"

for fasta in "$FASTA_DIR"/*.g.vcf
do
	echo "Indexing: $fasta"
        bcftools index "$fasta"
done

echo "Finished indexing all vcfs"
