**Step 1: Align assemblies using PHG commands**

Necessary files:
- reference fasta
- reference gff
- (mutated) assembly fastas


**Step 2: Pick crossover points in ref coordinates (create {assembly}_refkey.bed files)**

`python pick_crossovers.py`

User must define assembly names and chromosomes


**Step 3: Create chain files (create {assembly}.chain) from maf files**

`bash create_chains.sh -i maf_file_directory -o chain_file_directory -j 12`


**Step 4: Convert ref coordinates to assembly coordinates (create {assembly}_key.bed files)**

`python convert_coords.py`


**Step 5: Generate recombined sequences (create {founder}_key.bed and {founder}.fa files)**

`python write_fastas.py`


**Step 6: Align recombined assemblies using PHG commands**

Necessary files:
- reference fasta
- reference gff
- (recombined) assembly fastas