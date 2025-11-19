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

**Step 9: Map fastq reads of the original, non-mutated and non-recombined assemblies**

`ropebwt mem`