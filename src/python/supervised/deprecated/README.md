# README

This deprecated directory contains various experimental scripts for 
training ML models to identify the presence and/or position of recombination
sites/crossovers from PS4G data (reformatted to numpy format). None of these
models were especially successful, but we're keeping the scripts as a record
of what was done and so that the code can be revisited if needed for another
purpose. 

### Files included
- dataset_labeled.py: contains dataset classes formatted for the various models. 
- models_labeled.py: custom pytorch model architectures
- loss_functions.py: custom loss functions and wrappers
- simulate_learnable_data.py: simulates datasets containing crossovers for troubleshooting purposes
- generate_keyfile.py: all datasets in dataset_labeled require a keyfile generated with this script
- subset_windows.py: all datasets optionally take a set of windows for training. This script performs downsampling and outputs the windows file.
- train_*.py: training scripts for the various model architectures tested
- eval_model.py: combined evaluation script. Should work with any model trained with train_*.py


### Example pipeline

1. acquire training data courtesy of Sarah's pipeline, in .npy format
2. split into a training directory and a testing directory.
3. generate keyfile for training and testing sets:

    `python generate_keyfile.py --dir data/training/ --output keyfile.tsv`

    `python generate_keyfile.py --dir data/testing/ --output keyfile_test.tsv`
4. optionally, downsample windows that have no crossovers:
   
   4a. for next crossover prediction:

    `python subset_windows.py --keyfile keyfile.tsv --output windows.txt --input-len 256 --step-size 32 --no-crossover-rate 0.1`

    4b. for crossover detection (binary classification):

    `python subset_windows.py --keyfile keyfile.tsv --output windows.txt --input-len 32 --step-size 8 --padding 8 --no-crossover-rate 0.05 --border-retain-rate 0.2`
5. run training script
   
    5a. `python train_decoder_only.py --input-len 256 --step-size 32 --keyfile keyfile.tsv --windows windows.txt --num-epochs 5 --run-name test-run --batch-size 32 --save-model-path test_run_0/`

    5b. `python train_CNN.py --step-size 8 --keyfile keyfile.tsv --windows windows.txt --mode binary --padding 8 --conv 2d --num-epochs 5 --run-name test-run --project-name CNN-training --batch-size 256 --save-model-path test_run_0/`
6. evaluate
   
    `python eval_model.py --input-len 256 --step-size 128 --checkpoint /full/path/to/test_run_0/ --keyfile keyfile_test.tsv --output output.tsv --model-type decoder`

The resulting output.tsv is a table of the predicted and actual values for every testing window, plus the .npy file 
index and start position for reference purposes. 

### Additional Notes

We started with models that were inspired by segmentation analysis in computer vision - 
hence the use of pre-trained vision models. Since transformers are set up for token prediction,
we treated the output labels as discrete words in the vocabulary. This had the drawback that the relationship
between positions (i.e. that x is closer to x+1 than x+100) is not built in to the model or the loss function. 
In theory models can learn these sorts of relationships, but ours didn't on the training data we had available.

We tried implementing new loss functions that were "fuzzy" with respect to the correct label: one by combining
cross entropy losses at different resolutions (e.g. nearest 4 positions, nearest 8 positions), and one by
modeling our labels as binomial distributions centered around the true label and using Kullback-Leibler divergence loss.
Both could be made to work, but were significantly slower than cross-entropy loss. If the speed issue could be 
solved, I would like to revisit them. 

We also tried an LSTM for next crossover prediction, as this was easier to set up as a numerical
problem using mean-squared error loss. However, that left us with no special token to describe the end of 
decoding (when no more crossovers remain in the window)

After decoder-style prediction, we tried to build classifiers that could detect the
presence or absence of a crossover, without having to pinpoint the position. These used a smaller
context window, since a binary prediction would only be useful at a relatively small resolution. Tested 
architectures include pretrained ViT models, BERT for classification, CNNs, and even simple fully connected 
neural networks. These models learned more than the decoder models, but were not accurate
enough to be useful in the context of a larger imputation pipeline. 

I think that part of the challenge, especially with models that used a smaller context, is that
the input data is by nature very sparse. I think the models are having trouble distinguishing relevant
signal from 1. other biological patterns, and 2. genuine noise. There may be strategies to deal with 
these limitations - for example, different normalization or consensus strategies. Or, some of these
models may find use in other tasks. 


