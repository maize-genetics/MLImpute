# seq2seq imputation models

## Training

**sequence to sequence GRU models**

### Haploid

`python seq2seq.py
    --num-parents (number of parents)
    --max-seq-length (maximum input sequence length)
    --training-data-path (path to the input training data)
    --validation-data-path (path to the input validation data)
    --num-epochs (number of training epochs)
    --step-size (distance between the start points of each training window)
    --project-name (wandb project name)
    --run-name (wand run name)
    --batch-size (batch size)
    --save-model-path (path to save the best performing model)
    --steps-to-print (steps between reporting to wandb)
    --embedding-dim (embedding dimension)
    --hidden-dim (hidden dimension)
    --ls (smoothing hyperparameter)`

### Diploid

`python seq2seq_diploid.py
    --num-parents (number of parents)
    --max-seq-length (maximum input sequence length)
    --training-data-path (path to the input training data)
    --validation-data-path (path to the input validation data)
    --num-epochs (number of training epochs)
    --step-size (distance between the start points of each training window)
    --project-name (wandb project name)
    --run-name (wand run name)
    --batch-size (batch size)
    --save-model-path (path to save the best performing model)
    --steps-to-print (steps between reporting to wandb)
    --embedding-dim (embedding dimension)
    --hidden-dim (hidden dimension)
    --ls (smoothing hyperparameter)`



## Inference

### Haploid

`seq2seq_inference.py
    --data-path (path to the input data)
    --batch-size (batch size)
    --model-path (path to model)
    --save-dir (path to save imputed paths)
    --num-parents (number of parents)
    --max-seq-length (maximum input sequence length)
    --step-size (distance between the start points of each training window)
    --embedding-dim (embedding dimension)
    --hidden-dim (hidden dimension)`

### Diploid

`seq2seq_diploid_inference.py
    --data-path (path to the input data)
    --batch-size (batch size)
    --model-path (path to model)
    --save-dir (path to save imputed paths)
    --num-parents (number of parents)
    --max-seq-length (maximum input sequence length)
    --step-size (distance between the start points of each training window)
    --embedding-dim (embedding dimension)
    --hidden-dim (hidden dimension)`

## Visualization

If having trouble, comment out the assert statements or try to readjust the matrix dimensions. 

`../visualization/plot_predictions.py
    --sample (sample name)
    --chr (chromosome/contig name)
    --start (start position)
    --end (end position)
    --matrix-dir (path to matrix directory)
    --predictions (path to predictions numpy file)
    --diploid (add this flag for diploid predictions)`