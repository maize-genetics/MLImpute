# Training & Pipelines

This section collects the model-training and data-preparation workflows that
live alongside the GRITS Python package. The content below is included directly
from the per-module `README` files so it stays in sync with the source tree.

## Training-data / assembly pipeline

The cross-pipeline turns aligned assemblies into PS4G training matrices.

{%
   include-markdown "../src/python/cross/README.md"
   heading-offset=2
%}

## Seq2seq imputation models

Sequence-to-sequence GRU models for haploid and diploid imputation, plus
inference and visualization helpers.

{%
   include-markdown "../src/python/supervised/README.md"
   heading-offset=2
%}
