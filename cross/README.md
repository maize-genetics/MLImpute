**Step 1: Align assemblies using PHG commands**

Necessary files:
- reference fasta
- reference gff
- mutated assembly fastas (must be an even number of these)


**Step 2: Pick crossover points in ref coordinates (create {assembly}_refkey.bed files)**

`python pick_crossovers.py --ref-fasta /fastas-dir/ref.fa --assembly-list assembly_list.txt`

User must define assembly names and chromosomes


**Step 3: Create chain files (create {assembly}.chain) from maf files**

`bash create_chains.sh -i maf_file_directory -o chain_file_directory -j 12`


**Step 4: Convert ref coordinates to assembly coordinates (create {assembly}_key.bed files)**

`python convert_coords.py --assembly-list assembly_list.txt --chain-dir /path/to/chain-files/`


**Step 5: Generate recombined sequences (create {founder}_key.bed and {founder}.fa files)**

`python write_fastas.py --assembly-list assembly_list.txt --chromosome-list chromosome_list.txt --assembly-dir /path/to/asm_fastas/`

**Step 6: Reformat fasta files into lines with equal length**

`mkdir -p pretty_fastas
cd recombinate_fastas/
for f in *.fa; do
    seqkit seq -w 60 -j 8 "$f" > "../pretty_fastas/$f"
done`

**Step 7: Align recombined assemblies using PHG commands**

Necessary files:
- reference fasta
- reference gff
- recombined assembly fastas