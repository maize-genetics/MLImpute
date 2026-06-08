# seq2seq imputation models
Best architecture is currently seq2seq_diploid_joint_bert

(modernBERT encoder + GRU decoder with joint probability predictions)

run from the main grits dir use:

```python -m src.python.supervised.seq2seq_diploid_joint_bert```

## Training

**sequence to sequence GRU models**

### Haploid

```bash
python -m src.python.supervised.seq2seq
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
    --ls (smoothing hyperparameter)
```

### Diploid

```bash
python -m src.python.supervised.seq2seq_diploid
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
    --ls (smoothing hyperparameter)
```
- GRU encoder + GRU decoder

seq2seq_diploid_joint.py
- predicts a joint probability distribution rather than two independent preds

python seq2seq_diploid_joint_bert.py
- switches the GRU encoder for a modernBERT encoder

python seq2seq_diploid_joint_bert_pos.py
- an additional feature for position


## Inference

For inference only, matrices can be made with ps4g_to_matrix.py

```bash
python -m src.python.supervised.ps4g_to_matrix
    --ps4g-dir (directory containing PS4G matrices)
    --output-dir (output directory)
    --collapse (flag to collapse ps4g by position)
    --include-all-pos (flag to include empty positions, must collapse)
    --ref-fasta (path to reference fasta (required for --include-all-pos to obtain chr lengths))
 ```

### Haploid

```bash
python -m src.python.supervised.seq2seq_inference
    --data-path (path to the input data)
    --batch-size (batch size)
    --model-path (path to model)
    --save-dir (path to save imputed paths)
    --num-parents (number of parents)
    --max-seq-length (maximum input sequence length)
    --step-size (distance between the start points of each training window)
    --embedding-dim (embedding dimension)
    --hidden-dim (hidden dimension)
```

### Diploid

```bash
python -m src.python.supervised.seq2seq_diploid_inference
    --data-path (path to the input data)
    --batch-size (batch size)
    --model-path (path to model)
    --save-dir (path to save imputed paths)
    --num-parents (number of parents)
    --max-seq-length (maximum input sequence length)
    --step-size (distance between the start points of each training window)
    --embedding-dim (embedding dimension)
    --hidden-dim (hidden dimension)
```