**Step 1: Rename fasta files with phg prepare-assemblies**

- Need a keyfile with tab separated columns for filepaths and sample names
- Keep reference in a separate folder

**Step 2: Align assemblies using phg align-assemblies**

Necessary files:
- reference fasta
- reference gff
- mutated assembly fastas (must be an even number of these)


**Step 3: Pick crossover points in ref coordinates (create {assembly}_refkey.bed files)**

`python pick_crossovers.py --ref-fasta /fastas-dir/ref.fa --assembly-list assembly_list.txt`



**Step 4: Create chain files (create {assembly}.chain) from maf files**

`bash create_chains.sh -i maf_file_directory -o chain_file_directory -j 12`


**Step 5: Convert ref coordinates to assembly coordinates (create {assembly}_key.bed files)**

`python convert_coords.py --assembly-list assembly_list.txt --chain-dir /path/to/chain-files/`


**Step 6: Generate recombined sequences (create {founder}_key.bed and {founder}.fa files)**

`python write_fastas.py --assembly-list assembly_list.txt --chromosome-list chromosome_list.txt --assembly-dir /path/to/asm_fastas/`

**Step 7: Reformat fasta files into lines with equal length**

`mkdir -p pretty_fastas`

`cd recombinate_fastas/`

`for f in *.fa; do
    seqkit seq -w 60 -j 8 "$f" > "../pretty_fastas/$f"
done`

**Step 8: Align recombinate assemblies using phg align-assemblies**

Necessary files:
- reference fasta
- reference gff
- recombined assembly fastas

**Step 9: Convert maf files to gvcfs**

`biokotlin-tools maf-to-gvcf-converter --reference-file REF_FILE.fa --maf-file sample.maf --output-file sample.gvcf --sample-name sample`

**Step 10: Build spline knots**

`phg build-spline-knots --vcf-dir /path/to/gvcf-dir/ --vcf-type gvcf --output-dir splines`

**Step 11: Build ropebwt index (do not include the reference)**

`phg rope-bwt-chr-index --keyfile keyfile.txt --output-dir ropebwt-index --index-file-prefix ropebwt_index`

**Step 12: Map fastq reads of the original, non-mutated and non-recombined assemblies**

`ropebwt3 mem ropebwt-index/ropebwt_index.fmd sample.fq -l 148 -p num_parents*2 > ropebwt-output/sample.bed`

**Step 13: Convert ropebwt bed files into ps4g files**

`phg convert-ropebwt2ps4g-file --ropebwt-bed ropebwt-output/sample.bed --output-dir ps4g --spline-knot-dir splines --sort-positions`

**Step 14: Convert ps4g files into training matrices**

`python build_training_data.py --assembly-key-dir path/to/assembly_keys/ --ps4g-dir ps4g --output-dir training-data`