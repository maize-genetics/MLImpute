# train a VitForImageClasification model to detect whether or not a crossover occurs in the dataset

from transformers import get_wsd_schedule
from transformers import ViTForImageClassification
import torch
from transformers import Trainer, TrainingArguments
import argparse
import torch.optim as optim
from dataset_labeled import CategoricalVisionSegmentationDataset
import sys

# arguments for running the script
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-parents", type=int, default=24, help="number of parents")
    parser.add_argument("--image-size", type=int, default=384, help="image side length: should be multiple of num_parents")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to a previous training checkpoint")
    parser.add_argument("--keyfile", type=str, required=True, help="keyfile with list of input files")
    parser.add_argument("--windows", type=str, default=None, help="Optional, specify which windows to include")
    parser.add_argument("--num-epochs", "-e", type=int, default=2, help="number of training epochs")
    parser.add_argument("--skip-warmup", action="store_true", help="skip the warmup stage of WSD")
    parser.add_argument("--allow-cpu",  action="store_true", help="allow cpu for training (not recommended)")
    parser.add_argument("--run-name", "--rn", type=str, default="run-1", help="wandb run name")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="batch size")
    parser.add_argument("--save-model-path", "-s", type=str, default="output_model", help="path to save the best performing model")
    parser.add_argument("--loss-type", type=str, default="mean", help="type of categorical cross entropy loss to use. Choose from 'mean', 'sum' or 'fuzzy")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="max learning rate")
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        device = torch.device('cuda')
    else:
        if args.allow_cpu:
            device = torch.device('cpu')
            num_devices = 1
        else:
            print("Error: GPU not found")
            sys.exit()


    # for processing the "images"
    chunks = args.image_size // args.num_parents
    pos_length = args.image_size * chunks

    # WSD is suited for picking up from a previous training checkpoint
    # So, we make this an option
    if args.checkpoint is not None:
        model = ViTForImageClassification.from_pretrained(args.checkpoint)
    else:
        # If no checkpoint is provided, we initialize a new model
        # This could have been a parameter, but we hard-code it for now
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-384")

        # replace the last layer with a binary classifier
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)



    dataset = CategoricalVisionSegmentationDataset(args.keyfile, windows=args.windows)

    dataset_chunks = torch.utils.data.random_split(dataset, [0.9, 0.1])
    dataset_train = dataset_chunks[0]
    dataset_val = dataset_chunks[1]

    # set up model
    model = model.to(device)
    model.train()

    # set up optimizer and scheduler
    # we pre-calculate the number of training and warmup steps needed
    # based on batch size
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 10% warmup/decay
    len_warmup = (len(dataset_train) * args.num_epochs) // (args.batch_size * 10 * num_devices)
    if args.skip_warmup:
        lr_scheduler = get_wsd_schedule(optimizer, 0, len_warmup,
                                        num_training_steps=(args.num_epochs * len(dataset_train)) // (
                                                    args.batch_size * num_devices))
    else:
        lr_scheduler = get_wsd_schedule(optimizer, len_warmup, len_warmup,
                                        num_training_steps=(args.num_epochs * len(dataset_train)) // (
                                                args.batch_size * num_devices))

    # use huggingface trainer to avoid boilerplate code
    training_args = TrainingArguments(
        output_dir=args.save_model_path,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        report_to="wandb",
        run_name=args.run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        optimizers=(optimizer, lr_scheduler)
    )
    trainer.train()

if __name__ == '__main__':
    main()