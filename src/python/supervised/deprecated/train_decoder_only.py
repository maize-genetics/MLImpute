# train a ModernBertDecoder model for next crossover prediction

import torch
from transformers import get_wsd_schedule
import argparse
import sys
from transformers import Trainer, TrainingArguments
import torch.optim as optim
from models_labeled import DecoderOnlyConfig, DecoderOnlyModel
from dataset_labeled import BaseSegmentationDataset
from loss_functions import CrossEntropy, BinomialKLLoss


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-parents", type=int, default=24, help="number of parents")
    parser.add_argument("--step-size", type=int, default=128, help="step size between window starts")
    parser.add_argument("--input-len", type=int, default=256, help="length of input ps4g matrix")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to a previous training checkpoint")
    parser.add_argument("--keyfile", type=str, required=True, help="keyfile with list of input files")
    parser.add_argument("--windows", type=str, default=None, help="Optional, specify which windows to include")
    parser.add_argument("--num-epochs", "-e", type=int, default=2, help="number of training epochs")
    parser.add_argument("--distribute-loss", action="store_true",
                        help="use distributed loss instead of cross-entropy loss")
    parser.add_argument("--skip-warmup", action="store_true", help="skip the warmup stage of WSD")
    parser.add_argument("--allow-cpu", action="store_true", help="allow cpu for training (not recommended)")
    parser.add_argument("--run-name", "--rn", type=str, default="run-1", help="wandb run name")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="batch size")
    parser.add_argument("--save-model-path", "-s", type=str, default="output_model",
                        help="path to save the best performing model")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="max learning rate")

    parser.add_argument("--save-steps", type=int, default=2000, help="number of steps between saved checkpoints")
    parser.add_argument("--eval-steps", type=int, default=2000, help="number of steps between evaluations")
    parser.add_argument("--logging-steps", type=int, default=100, help="number of steps between wandb logs")
    # TODO: allow more control over fuzzy loss
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
    pos_length = args.input_len
    step_size = args.step_size

    # WSD is suited for picking up from a previous training checkpoint
    # So, we make this an option
    if args.checkpoint is not None:
        model = DecoderOnlyModel.from_pretrained(args.checkpoint)
    else:
        # If no checkpoint is provided, we initialize a new model
        config = DecoderOnlyConfig(num_parents=args.num_parents, vocab_size=pos_length + 3,
                                   max_position_embeddings=pos_length, is_decoder=True, pad_token_id=pos_length,
                                   eos_token_id=pos_length + 1, bos_token_id=pos_length + 2,
                                   cls_token_id=pos_length + 1, sep_token_id=pos_length + 2, add_cross_attention=True,
                                   num_hidden_layers=6, num_attention_heads=12)

        model = DecoderOnlyModel(config)

    if args.distribute_loss:
        loss_weights = torch.ones(pos_length + 3, dtype=torch.float)
        loss_weights[pos_length:pos_length + 3] = 0.01  # weight special tokens less than regular tokens
        dataset = BaseSegmentationDataset(args.keyfile, windows=args.windows, input_len=pos_length, step_size=step_size,
                                          preload=True)
        criterion = CrossEntropy(reduction="sum", weight=loss_weights)
    else:
        dataset = BaseSegmentationDataset(args.keyfile, windows=args.windows, input_len=pos_length, step_size=step_size,
                                          preload=True, distribute_label_density=True)
        criterion = BinomialKLLoss(reduction="sum")

    model.model._loss_function = criterion

    dataset_chunks = torch.utils.data.random_split(dataset, [0.95, 0.05])
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
        eval_steps=args.eval_steps,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=args.logging_steps,
        report_to="wandb",
        save_steps=args.save_steps,
        run_name=args.run_name
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=dataset.collate,
        optimizers=(optimizer, lr_scheduler)
    )
    trainer.train()


if __name__ == '__main__':
    main()
