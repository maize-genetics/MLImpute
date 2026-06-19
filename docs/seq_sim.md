# Sequence Simulation (seq_sim)

[`seq_sim`](https://github.com/maize-genetics/seq_sim) is the accompanying DNA
sequence-simulation pipeline for GRITS. It is a Kotlin/Gradle application that
performs assembly alignment, variant simulation, recombination, and conversion
to the PS4G format consumed by the GRITS imputation models.

The pages in this section are **synced automatically** from the `seq_sim`
repository at build time by `scripts/fetch-external-docs.sh`. Edit them in the
upstream repo, not here.

![seq_sim pipeline overview](_external/seq_sim/images/grits_v2_seq_sim_pipeline.svg){ loading=lazy }

## Upstream README

{%
   include-markdown "_external/seq_sim/README.md"
   heading-offset=1
%}
