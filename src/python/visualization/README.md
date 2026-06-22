## Visualization

If having trouble, there could be bugs with matrix dimensions and haploid/diploid options

```bash
python -m src.python.visualization.plot_predictions
    --sample (sample name)
    --chr (chromosome/contig name)
    --start (start position)
    --end (end position)
    --matrix-dir (path to matrix directory)
    --predictions (path to predictions numpy file)
    --diploid (add this flag for diploid predictions)
```