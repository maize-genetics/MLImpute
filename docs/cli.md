This comprehensive guide covers the command-line interface (CLI) for GRITS, a machine learning-based haplotype imputation tool that combines multiple approaches including BiMamba, ModernBERT, HMM-based methods, and K-Nearest Neighbors.

## Installation

### Prerequisites
- Python 3.10 or higher
- [pixi](https://pixi.sh/latest/) (recommended)
- CUDA 12+ (optional, for GPU acceleration)

### Using pixi (Recommended)

```bash
# Clone the repository
git clone https://github.com/maize-genetics/grits.git
cd grits

# CPU-only installation
pixi install

# GPU-enabled installation (Linux only)
pixi install --environment gpu
```


### Verify Installation

```bash
# Test the CLI with help
pixi run -- python src/impute.py --help
```

## Quick Start

Basic imputation with default settings:

```bash
# Using pixi
pixi run -- python src/impute.py --input data.ps4g --output results.bed --model bimamba

# Using conda
python src/impute.py --input data.ps4g --output results.bed --model bimamba
```

## Command-Line Interface

### Basic Syntax

```bash
python src/impute.py [OPTIONS]
```

### Required Arguments

| Argument   | Description                                                | Example                        |
|------------|------------------------------------------------------------|--------------------------------|
| `--input`  | Input PS4G file path                                       | `--input data/sample.ps4g`     |
| `--output` | Output BED file path                                       | `--output results/imputed.bed` |
| `--model`  | Model to use for imputation (_see next table for options_) | `--model bimamba`              |

### Available Models

| Model         | Description                                 | GPU Support | Best Use Case                                 |
|---------------|---------------------------------------------|-------------|-----------------------------------------------|
| `bimamba`     | State space model for sequence imputation   | ✅           | Large datasets with complex patterns          |
| `modernbert`  | Transformer-based genetic sequence modeling | ✅           | High-accuracy imputation tasks                |
| `knn`         | K-Nearest Neighbors baseline method         | ❌           | Small datasets, baseline comparisons          |

### Optional Arguments

| Argument           | Default  | Description                                                                                                                                                                                                                                                           |
|--------------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--weight`         | `global` | Weighting strategy used when converting PS4G data into numerical matrices.                                                                                                                                                                                            |
| `--collapse`       | `false`  | Collapse gamete sets in processing.                                                                                                                                                                                                                                   |
| `--verbose`        | `false`  | Enable verbose logging.                                                                                                                                                                                                                                               |
| `--global-weights` | `none`   | Specifies an external file containing precomputed global weight values to be used during imputation and (optionally) for HMM transition scaling.                                                                                                                      |
| `--HMM`            | `false`  | Enable Viterbi decoding with a Hidden Markov Model (HMM) on top of the ModernBERT or Bi-Mamba emissions to enforce sequence smoothness and reduce spurious switches between parents/states.                                                                           |
| `--diploid`        | `false`  | Enables diploid imputation mode. Predicts two alleles (parent pairs) per site instead of a single haploid call. Uses either top-2 class probabilities (ModernBERT-only) or pair-state Viterbi decoding when `--HMM` is active.                                        |
| `--window-size`    | `1000`   | Size of the sliding window for sequence segmentation. Defines the number of loci processed together per forward pass in ModernBERT or Bi-Mamba. Smaller windows increase granularity but may reduce context; larger windows improve continuity at higher memory cost. |
| `--collapse-bed`   | `false`  | Collapses contiguous BED intervals with identical parental calls into single merged regions in the output file. Useful for reducing file size and improving readability of imputed BED outputs.                                                                       |


## Model-Specific Usage

### BiMamba
This section provides examples and explanations for running the `impute.py` CLI using the **BiMamba** model backend. The BiMamba imputation model performs efficient state-space sequence modeling for haplotype reconstruction.

#### Basic (haploid) BiMamba run

```bash
python impute.py \
  --input maize_data.ps4g \
  --output imputed_maize.bed \
  --model mamba \
  --weight global \
  --window-size 1000
```

> **Details:**
> 
> * `--model mamba` selects the BiMamba backend.
> * Uses **global weighting** derived from PS4G file for parental confidence.
> * Imputation is done per window of 1000 loci using BiMamba’s learned state-space model.
> * Outputs a standard haploid BED file.


#### BiMamba with HMM smoothing

```bash
python impute.py \
  --input maize_data.ps4g \
  --output imputed_maize_HMM.bed \
  --model mamba \
  --weight global \
  --global-weights weights.npy \
  --HMM \
  --window-size 1000
```

> **Details:**
> 
> * `--HMM` enables a **Hidden Markov Model decoder** that smooths transitions between parent states.
> * `--global-weights weights.npy` supplies precomputed global weights that influence **stay vs switch probabilities** (`p_stay = max(weights) × 0.20`).
> * Produces smoother haplotype segments with fewer parent switches.
> * Ideal for long chromosomes or noisy genotype data.


#### Diploid BiMamba imputation

```bash
python impute.py \
  --input maize_data.ps4g \
  --output imputed_maize_diploid.bed \
  --model mamba \
  --diploid \
  --weight global \
  --window-size 1000
```

> **Details:**
> 
> * `--diploid` switches to diploid inference mode.
> * BiMamba outputs pairs of parental indices `(p1, p2)` at each locus.
> * Internally uses a pair-state Viterbi structure when `--HMM` is active.

---

#### Diploid BiMamba with HMM smoothing

```bash
python impute.py \
  --input maize_data.ps4g \
  --output imputed_maize_diploid_HMM.bed \
  --model mamba \
  --diploid \
  --HMM \
  --global-weights weights.npy \
  --window-size 1000 \
  --collapse-bed
```

> **Details:**
> 
> * Combines both flags: `--diploid` + `--HMM`.
> * HMM operates over **pair states** to enforce biologically realistic allele transitions (max one allele change per step).
> * `--collapse-bed` merges contiguous identical parent-pair regions for a compact BED file.
> * Best suited for full chromosome-scale reconstruction or visualization.

> [!NOTE]
> 
> **GPU requirement:** BiMamba requires CUDA; if unavailable, `impute.py` raises an error:
> 
>   ```python
>   if not torch.cuda.is_available():
>       raise EnvironmentError("CUDA is not available. BiMamba requires a GPU to run.")
>   ```


### ModernBERT
This section provides examples and explanations for running the `impute.py` CLI using the **ModernBERT** model backend. The ModernBERT imputation model is a Transformer-based sequence encoder optimized for genomic imputation tasks.

#### Basic (haploid) ModernBERT run

```bash
python impute.py \
  --input maize_data.ps4g \
  --output imputed_maize.bed \
  --model modernbert \
  --weight global \
  --window-size 1000
```

> **Details:**
> 
> * `--model modernbert` selects the ModernBERT backend.
> * Uses **global weighting** from the PS4G input file to inform parent confidence.
> * Performs per-window sequence encoding and classification to assign parent labels.
> * Outputs a haploid BED file containing one parent call per locus.


#### ModernBERT with HMM smoothing

```bash
python impute.py \
  --input maize_data.ps4g \
  --output imputed_maize_HMM.bed \
  --model modernbert \
  --weight global \
  --global-weights weights.npy \
  --HMM \
  --window-size 1000
```

> **Details:**
> 
> * `--HMM` enables the **Hidden Markov Model** decoder that smooths local predictions across windows using Viterbi decoding.
> * `--global-weights weights.npy` provides precomputed global weights to influence transition probabilities (`p_stay = max(weights) × 0.20`).
> * The resulting imputation exhibits longer, more continuous haplotype segments.


#### Diploid ModernBERT imputation

```bash
python impute.py \
  --input maize_data.ps4g \
  --output imputed_maize_diploid.bed \
  --model modernbert \
  --diploid \
  --weight global \
  --window-size 1000
``` 

> **Details:**
> 
> * Enables **diploid imputation** mode, producing two parental calls `(p1, p2)` per site.
> * Without HMM, diploid predictions are inferred from **top-2 class probabilities** of ModernBERT’s output logits.
> * Heterozygosity is determined when the probability ratio between the top-2 parents exceeds a set threshold (default: `ratio > 0.8`).


#### Diploid ModernBERT with HMM smoothing

```bash
python impute.py \
  --input maize_data.ps4g \
  --output imputed_maize_diploid_HMM.bed \
  --model modernbert \
  --diploid \
  --HMM \
  --global-weights weights.npy \
  --window-size 1000 \
  --collapse-bed
```

> **Details:**
>
> * Combines **diploid inference** with **HMM-based smoothing**.
> * The HMM operates over **pair states**, penalizing simultaneous double-switches between parent pairs.
> * `--collapse-bed` merges contiguous BED intervals with identical parent pairs for compact output.
> * Provides the most biologically realistic imputation results for full-chromosome analyses.

> [!NOTE]
>
> **GPU requirement:** GPU machines are highly recommended for the ModernBERT model approach.


### KNN
This section provides examples and explanations for running the `impute.py` CLI using the **K-Nearest Neighbors (KNN)** model backend. The KNN model provides a simple, interpretable baseline for haplotype imputation by leveraging local genotype similarity across parental samples.

#### Basic KNN imputation

```bash
python impute.py \
  --input maize_data.ps4g \
  --output imputed_maize_knn.bed \
  --model knn \
  --weight global \
  --window-size 21
```

> **Details:**
> 
> * `--model knn` selects the KNN backend.
> * Uses **global weighting** from the PS4G file to account for parental confidence.
> * `--window-size` defines the number of neighboring loci used for majority voting (must be an odd number).
> * Outputs a haploid BED file with predicted parent indices per locus.


#### Unweighted KNN run

```bash
python impute.py \
  --input maize_data.ps4g \
  --output imputed_maize_knn_unweighted.bed \
  --model knn \
  --weight unweighted \
  --window-size 21
```

> **Details:**
> 
> * `--weight unweighted` disables weighting, treating all parents equally during KNN voting.
> * Useful for testing or when no reliable parental priors are available.


#### Diploid KNN imputation

```bash
python impute.py \
  --input maize_data.ps4g \
  --output imputed_maize_knn_diploid.bed \
  --model knn \
  --diploid \
  --weight global \
  --window-size 21
```

> **Details:**
> 
> * `--diploid` enables prediction of **two parents (p1, p2)** per site instead of one.
> * The KNN model independently predicts two most probable parents per position based on voting frequency.
> * Suitable for heterozygous or mixed parental contributions.


#### Additional Notes

* **Window size constraint:** Must be **odd** to ensure a tiebreaker in majority voting.
* **Computation:** KNN is CPU-based and lightweight, suitable for quick runs or baseline benchmarking.



## File Formats
This section provides a high-level explanation of the **PS4G** input 
file and the **BED** output files used by the imputation CLI 
application. These formats define how genotype data is read, processed, 
and written during imputation.


### PS4G (Input)

#### Purpose
The PS4G file encodes **parental genotype observations** across loci 
and is the main input for all imputation models (`KNN`, `BiMamba`, 
`ModernBERT`). It provides the parental index mapping, variant 
coordinates, and weighting information used during processing.

#### File Type
Plain-text, **tab-separated values (TSV)** file.

#### Structure Overview

* Lines beginning with `#` represent **metadata**, and appear only in a
  leading block at the top of the file.
* Within that leading block, the `#gamete\tgameteIndex\tcount` tag line
  opens a **gamete section**: every `#`-prefixed line after it (other than
  a keyed metadata line like `#TotalUniqueCounts:`) is a gamete record,
  until the block ends.
* The un-prefixed `gameteSet\trefContig\trefPosBinned\tcount` column
  header ends the leading block, and everything after it is **locus data**.
  `#`-prefixed lines appearing after this point are not read as metadata or
  gamete records — they're inert trailing comments.

#### Example PS4G File
Below is a minimal example of a PS4G input file used for testing:

```
#PS4G
#version=2.0
#TotalUniqueCounts: 4
#gamete	gameteIndex	count
#B73	0	4
#CML247	1	2
#W22	2	1
gameteSet	refContig	refPosBinned	count
0	chr1	512000	1
0,1	chr1	512256	1
0	chr1	512512	1
0,1,2	chr1	512768	1
```

#### Metadata Lines (`#` prefix)

Metadata defines the parental gamete set and read-depth weights.

Gamete records are recognized **by position, not by column shape**: a
line is a gamete record only if it appears between the `#gamete` tag and
the end of the leading `#`-line block. The tag is matched on its first
tab field only (case-insensitively), so the column names after it
(`gameteIndex`, `count`) are purely informational. A file with no
`#gamete` tag declares no gamete records from its header — the loader and
viewer instead derive one gamete per distinct index found in the data
section's `gameteSet` column, named by that index (see below).

The `gamete` field itself may be written either as a bare sample name
(`B73`) or with an explicit haplotype/gamete index suffix (`B73:0`); a
bare name implies index `0`. Both forms are valid per the PS4G spec and
are accepted identically by the loader (`ps4g_io/ps4g.py`) and the
desktop/web viewer (`parser-core/src/ps4g.rs`).

Each metadata entry contains:

| Field          | Description                                         |
|----------------|-----------------------------------------------------|
| `gamete`       | Parent identifier (e.g., B73, CML247, W22).         |
| `gamete_index` | Integer index assigned to the parent.               |
| `read_count`   | Total read count or coverage supporting the parent. |

From the example above:

| Gamete | Index | Read Count | Weight (% of Reads) | % of Hits |
|--------|-------|------------|----------------------|-----------|
| B73    | 0     | 4          | 1.000                | 0.571     |
| CML247 | 1     | 2          | 0.500                | 0.286     |
| W22    | 2     | 1          | 0.250                | 0.143     |

A **global weight** is derived as:

```python
weight = read_count / total_reads
```

`total_reads` is the sum of the `count` column over all data rows (4 in
the example above) — **not** the sum of the per-gamete `read_count`
values (4 + 2 + 1 = 7, the "% of Hits" column). A read whose `gameteSet`
names several gametes is credited to each of them, so that sum counts
reads more than once; weights normalized against it (`% of Hits`) always
sum to 1, while `weight`/`% of Reads` need not.

The desktop/web viewer (`parser-core/src/ps4g.rs`) recomputes `total_reads`
from the data section rather than trusting the `#TotalUniqueCounts` header,
since that value is producer-declared and not always a read count — the
CRF window exporter (`crf/export_windows.py`), for instance, writes the sum
of all (site, founder) hits there instead.

Weights can be ignored when using `--weight unweighted`.

#### Data Lines (loci rows)

Each row encodes one genomic locus with its coordinate and contributing parent indices.

Example (from the test file):

```text
0,1,2	chr1	512768	1
```

| Column         | Description                                                   |
|----------------|---------------------------------------------------------------|
| `gameteSet`    | Comma-separated indices of parents contributing to the locus. |
| `refContig`    | Chromosome/contig identifier (e.g., "chr1", "chr2").          |
| `refPosBinned` | Binned genomic position in base pairs.                        |
| `count`        | Observation count or coverage for this site.                  |

#### Position Identifiers

Internally, the PS4G processing code creates unique position identifiers by combining the chromosome and position:

```python
position_id = f"{refContig}_{refPosBinned}"
```

For example, `chr1_512768` uniquely identifies a locus on chromosome 1 at position 512768.

#### What the Loader Produces

`convert_ps4g(ps4g_file, weight, collapse)` returns:

* **`input_matrix`** - binary (or float) matrix `[num_sites, num_parents]` where entries indicate parent presence.
* **`weights`** - vector of per-parent weights (from metadata or uniform).

These outputs are consumed by model-specific data loaders such as `WindowIndexDatasetFromMatrix`.



### BED File (Output)

#### Purpose

The BED output stores imputed parental assignments per locus (haploid or diploid). Output format depends on whether collapsing of identical regions is enabled.

#### A) Uncollapsed (per-site) BED Format

Each row corresponds to a single imputed site.

| Column    | Description                                                          |
|-----------|----------------------------------------------------------------------|
| `chrom`   | Chromosome/contig identifier (e.g., "chr1", "chr2").                 |
| `pos`     | Genomic coordinate in base pairs (from `refPosBinned`).              |
| `parent1` | Predicted parent (or allele 1).                                      |
| `parent2` | Predicted parent (or allele 2; may equal `parent1` in haploid mode). |

Example:

```text
chr1	512000	B73	B73
chr1	512256	B73	CML69
chr1	512512	B73	CML69
```

#### B) Collapsed BED Format

When `--collapse-bed` is set, contiguous sites with identical predictions are merged into intervals.

| Column    | Description                                      |
|-----------|--------------------------------------------------|
| `chrom`   | Chromosome/contig identifier (e.g., "chr1").     |
| `start`   | Start coordinate of merged interval (in bp).     |
| `end`     | End coordinate of merged interval (in bp).       |
| `parent1` | First predicted parent.                          |
| `parent2` | Second predicted parent.                         |

Example:

```text
chr1	512000	512768	B73	CML69
```

This reduces redundancy and yields a compact representation of parental segments.



