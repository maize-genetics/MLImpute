**Step 1: Align assemblies using PHG commands**

Necessary files:
- reference fasta
- reference gff
- (mutated) assembly fastas


**Step 2: Pick crossover points in ref coordinates (create {assembly}_refkey.bed files)**

`python pick_crossovers.py`

User must define assembly names and chromosomes


**Step 3: Create chain files (create {assembly}.chain) from maf files**

`./create_chains.sh`

**Step 4: Convert ref coordinates to assembly coordinates (create {assembly}_key.bed files)**



**Step 5: Assign unmapped assembly sequence (modify {assembly}_key.bed)**



**Step 6: Generate recombined sequences (create {founder}_key.bed and {founder}.fa files)**


**Step 7: Align recombined assemblies using PHG commands**