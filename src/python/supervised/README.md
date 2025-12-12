Instructions for running seq2seq models. 

The haploid and diploid versions are essentially the same, and in the future we can consolidate. 



**Haploid sequence to sequence GRU model:**

python seq2seq.py
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

seq2seq_inference.py
    --data-path (path to the input data)
    --batch-size (batch size)
    --model-path (path to model)
    --save-dir (path to save imputed paths)
    --num-parents (number of parents)
    --max-seq-length (maximum input sequence length)
    --step-size (distance between the start points of each training window)
    --embedding-dim (embedding dimension)
    --hidden-dim (hidden dimension)


**Diploid sequence to sequence GRU model:**

python seq2seq_diploid.py
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