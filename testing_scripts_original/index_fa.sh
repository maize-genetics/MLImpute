#!/bin/bash

# Folder containing FASTA files
FASTA_DIR="/workdir/irk9/data/phg-maize/gvcf-output"

for fasta in "$FASTA_DIR"/*.fa
do
	echo "Indexing: $fasta"
        bwa index "$fasta"
done

echo "Finished indexing all fastas"
