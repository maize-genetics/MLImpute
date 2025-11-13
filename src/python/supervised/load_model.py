from tqdm import tqdm
from transformers import VisionEncoderDecoderModel
import torch
import argparse
from dataset_vision import SegmentationDataset, custom_data_collator, SumCrossEntropy, BinnedCrossEntropy
import sys
from torch.utils.data import DataLoader

# arguments for running the script
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-parents", type=int, default=24, help="number of parents")
    parser.add_argument("--image-size", type=int, default=384, help="image side length: should be multiple of num_parents")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to a previous training checkpoint")
    parser.add_argument("--keyfile", type=str, required=True, help="keyfile with list of input files")
    parser.add_argument("--allow-cpu",  action="store_true", help="allow cpu for training (not recommended)")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="batch size")
    args = parser.parse_args()
    return args


args = parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    if args.allow_cpu:
        device = torch.device('cpu')
    else:
        print("Error: GPU not found")
        sys.exit()

# for processing the "images"
chunks = args.image_size // args.num_parents
pos_length = args.image_size * chunks

# WSD is suited for picking up from a previous training checkpoint
# So, we make this an option

model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint)

dataset = SegmentationDataset(args.keyfile, include_index=True)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_data_collator)

dl = iter(dataloader)

model.eval()

batch = next(dl)

model._loss_function = SumCrossEntropy()

outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"],
                decoder_input_ids=batch["decoder_input_ids"], decoder_attention_mask=batch["decoder_attention_mask"])

c1 = BinnedCrossEntropy(0, 6144)

loss1 = c1(outputs["logits"], batch["labels"], 0, 0)