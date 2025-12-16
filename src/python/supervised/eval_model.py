# Script to evaluate model on a held-out dataset
import math

from tqdm import tqdm
from transformers import ViTForImageClassification, VisionEncoderDecoderModel
import torch
import argparse
from src.python.supervised.deprecated import dataset_labeled, models_labeled
import sys
from torch.utils.data import DataLoader
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-parents", type=int, default=24, help="number of parents")
    parser.add_argument("--input-len", type=int, default=6144, help="length of input matrix")
    parser.add_argument("--step-size", type=int, default=1536, help="the offset between input windows")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to a previous training checkpoint")
    parser.add_argument("--keyfile", type=str, required=True, help="keyfile with list of input files")
    parser.add_argument("--allow-cpu", action="store_true", help="allow cpu for training (not recommended)")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="batch size")
    parser.add_argument("--output", type=str, required=True, help="path to output file")
    parser.add_argument("--model-type", type=str, default="vit-class", choices=["vit-class", "vit-decoder", "decoder", "lstm", "cnn-binary", "cnn-multiclass", "fully-connected"],
                        help="type of model. One of: vit-class, vit-decoder, decoder, lstm, cnn-binary, cnn-multiclass, fully-connected")
    parser.add_argument("--split-norm-levels", action="store_true",
                        help="for ViT models: whether to use local, chromosomal, and global normalization as the RGB channels")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        if args.allow_cpu:
            device = torch.device('cpu')
        else:
            print("Error: GPU not found")
            sys.exit()

    # load appropriate model and dataset
    if args.model_type == "vit-class":
        input_name = "pixel_values"
        image_size = int(math.sqrt(args.input_len * args.num_parents))
        model = ViTForImageClassification.from_pretrained(args.checkpoint)
        dataset = dataset_labeled.CategoricalVisionSegmentationDataset(args.keyfile, image_size=image_size,
                                                                       num_parents=args.num_parents,
                                                                       step_size=args.step_size,
                                                                       split_norm_levels=args.split_norm_levels,
                                                                       preload=True,
                                                                       include_index=True)
    elif args.model_type == "vit-decoder":
        input_name = "pixel_values"
        image_size = int(math.sqrt(args.input_len * args.num_parents))
        model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint)
        dataset = dataset_labeled.VisionSegmentationDataset(args.keyfile, image_size=image_size,
                                                            num_parents=args.num_parents,
                                                            step_size=args.step_size,
                                                            split_norm_levels=args.split_norm_levels,
                                                            preload=True,
                                                            include_index=True)
    elif args.model_type == "decoder":
        input_name = "encoder_hidden_states"
        model = models_labeled.DecoderOnlyModel.from_pretrained(args.checkpoint)
        dataset = dataset_labeled.BaseSegmentationDataset(args.keyfile, input_len=args.input_len,
                                                          num_parents=args.num_parents,
                                                          step_size=args.steps_size,
                                                          preload=True,
                                                          include_index=True)
    elif args.model_type == "lstm":
        input_name = "encoder_hidden_states"
        model = models_labeled.LstmSegmentationEncDec(args.num_parents, args.input_len, 12, 0.1)

        state_dict = torch.load(args.checkpoint)
        model.load_state_dict(state_dict)

        dataset = dataset_labeled.BaseSegmentationDataset(args.keyfile, input_len=args.input_len,
                                                          num_parents=args.num_parents,
                                                          step_size=args.steps_size,
                                                          preload=True,
                                                          include_index=True,
                                                          end_token=args.input_len * 4)
    elif args.model_type == "cnn-binary":
        input_name = "inputs_embeds"
        model = models_labeled.CNN()

        state_dict = torch.load(args.checkpoint)
        model.load_state_dict(state_dict)

        dataset = dataset_labeled.CategoricalDataset(args.keyfile, input_len=args.input_len,
                                                     step_size=args.step_size,
                                                     num_parents=args.num_parents,
                                                     preload=True,
                                                     include_index=True)
    elif args.model_type == "fully-connected":
        input_name = "inputs_embeds"
        model = models_labeled.FullyConnected()

        state_dict = torch.load(args.checkpoint)
        model.load_state_dict(state_dict)

        dataset = dataset_labeled.CategoricalDataset(args.keyfile, input_len=args.input_len,
                                                     step_size=args.step_size,
                                                     num_parents=args.num_parents,
                                                     preload=True,
                                                     include_index=True)
    else:
        input_name = "inputs_embeds"
        model = models_labeled.CNN(output_dim=(args.input_len // 2 + 1), sigmoid=False)

        state_dict = torch.load(args.checkpoint)
        model.load_state_dict(state_dict)

        dataset = dataset_labeled.CategoricalMulticlassDataset(args.keyfile, input_len=args.input_len,
                                                               step_size=args.step_size,
                                                               num_parents=args.num_parents,
                                                               preload=True,
                                                               include_index=True)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate)

    if args.model_type == "vit-class" or args.model_type == "cnn-binary":
        binary = True
    else:
        binary = False

    if args.model_type == "decoder" or args.model_type == "vit-decoder":
        generate = True
    else:
        generate = False

    # set up model
    model = model.to(device)
    model.eval()

    file_idx = [0] * len(dataset)
    start_pos = [0] * len(dataset)
    actual = [0] * len(dataset)
    predicted = [0] * len(dataset)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating..."):
            input_ids = batch[input_name].to(device)

            file_indices = batch["file_idx"]
            start_positions = batch["start_pos"] * args.step_size
            labels = batch["labels"]

            indices = batch["idx"]

            if generate:
                outputs = model.generate(input_ids).cpu().detach()
            else:
                outputs = model(input_ids).cpu().detach()

            file_idx[indices[0]:indices[-1]+1] = file_indices.detach().tolist()
            start_pos[indices[0]:indices[-1] + 1] = start_positions.detach().tolist()

            if binary:
                labels = [str(lab) for lab in labels]
                output = [str(out) for out in outputs]
            else:
                labels = [",".join([str(lb) for lb in lab]) for lab in labels]
                output = [",".join([str(op) for op in out if op < args.input_len]) for out in outputs]

            actual[indices[0]:indices[-1]+1] = labels
            predicted[indices[0]:indices[-1]+1] = output

    df = pd.DataFrame({"file_idx": file_idx,
                       "start_pos": start_pos,
                       "actual": actual,
                       "predicted": predicted})

    df.to_csv(args.output, sep="\t")


if __name__ == '__main__':
    main()
