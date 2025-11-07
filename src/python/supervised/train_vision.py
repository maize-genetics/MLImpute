from transformers import BertConfig, VisionEncoderDecoderModel, get_wsd_schedule
from transformers import ViTModel, BertLMHeadModel
import torch
from transformers import Trainer, TrainingArguments
import argparse
import torch.optim as optim
from dataset_vision import SegmentationDataset, custom_data_collator

 # TODO NOTE: this version contains a yet-unsolved bug. Will update when fixed

# arguments for running the script
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-parents", type=int, default=24, help="number of parents")
    parser.add_argument("--image-size", type=int, default=384, help="image side length: should be multiple of num_parents")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to a previous training checkpoint")
    parser.add_argument("--input-files", type=str, required=True, help="comma-separated list of input files")
    parser.add_argument("--num-epochs", "-e", type=int, default=2, help="number of training epochs")
    parser.add_argument("--run-name", "--rn", type=str, default="run-1", help="wandb run name")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="batch size")
    parser.add_argument("--save-model-path", "-s", type=str, default="output_model", help="path to save the best performing model")

    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # for processing the "images"
    chunks = args.image_size // args.num_parents
    pos_length = args.image_size * chunks

    # WSD is suited for picking up from a previous training checkpoint
    # So, we make this an option
    if args.checkpoint is not None:
        model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint)
    else:
        # If no checkpoint is provided, we initialize a new model
        config_decoder = BertConfig(vocab_size=pos_length + 3, max_position_embeddings=8192, is_decoder=True, pad_token_id=pos_length, eos_token_id=pos_length + 1,
                                          bos_token_id=pos_length + 2, cls_token_id=pos_length + 1, sep_token_id=pos_length + 2, add_cross_attention=True)

        # This could have been a parameter, but we hard-code it for now
        encoder = ViTModel.from_pretrained("google/vit-base-patch16-384")
        decoder = BertLMHeadModel(config_decoder)

        model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

        # Some of the token ID's don'r propagate properly from the decoder, so set again here
        model.config.decoder_start_token_id = pos_length+2
        model.config.pad_token_id = pos_length
        model.config.vocab_size = pos_length+3
        model.config.eos_token_id = pos_length+1

        # model.loss_function(NLLLoss())


    # set up dataset, including random split for validation
    # TODO: probably should replace with keyfile
    input_files = args.input_files.split(",")
    dataset = SegmentationDataset(input_files)

    dataset_chunks = torch.utils.data.random_split(dataset, [0.9, 0.1])
    dataset_train = dataset_chunks[0]
    dataset_val = dataset_chunks[1]

    # set up model
    model = model.to(device)
    model.train()

    # set up optimizer and scheduler
    # we pre-calculate the number of training and warmup steps needed
    # based on batch size
    # TODO: currently assumes that we'll be running on two gpus (e.g. on gpu08)
    optimizer = optim.AdamW(model.parameters())

    # 10% warmup/decay
    len_warmup = (len(dataset_train) * args.num_epochs) // (args.batch_size * 20)

    lr_scheduler = get_wsd_schedule(optimizer, len_warmup, len_warmup,
                                    num_training_steps=(args.num_epochs * len(dataset_train)) // (args.batch_size * 2))

    # use huggingface trainer to avoid boilerplate code
    training_args = TrainingArguments(
        output_dir=args.save_model_path,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        report_to="wandb",
        run_name=args.run_name
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=custom_data_collator,
        optimizers=(optimizer, lr_scheduler)
    )
    trainer.train()
