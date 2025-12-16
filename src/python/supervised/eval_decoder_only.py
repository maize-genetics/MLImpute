from tqdm import tqdm
import torch
import argparse
import sys
from src.python.supervised.deprecated.models_labeled import DecoderOnlyModel
from src.python.supervised.deprecated.dataset_labeled import BaseSegmentationDataset
from torch.utils.data import DataLoader

# arguments for running the script
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-parents", type=int, default=24, help="number of parents")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to a previous training checkpoint")
    parser.add_argument("--keyfile", type=str, required=True, help="keyfile with list of input files")
    parser.add_argument("--allow-cpu",  action="store_true", help="allow cpu for training (not recommended)")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="batch size")
    parser.add_argument("--output", type=str, required=True, help="path to output file")
    args = parser.parse_args()
    return args

def main():

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
    pos_length = 256

    # WSD is suited for picking up from a previous training checkpoint
    # So, we make this an option

    model = DecoderOnlyModel.from_pretrained(args.checkpoint)

    dataset = BaseSegmentationDataset(args.keyfile, include_index=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate)

    # set up model
    model = model.to(device)
    model.eval()



    with open(args.output, "w") as writer:
        writer.write("file_idx\tstart_pos\tactual\tpredicted\n")


        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating..."):
                input_ids = batch["encoder_hidden_states"].to(device)

                outputs = model.generate(encoder_hidden_states=input_ids, use_cache=False).cpu().detach()

                for idx in range(input_ids.shape[0]):
                    writer.write(str(int(batch["file_idx"][idx])) + "\t" + str(int(batch["pos_idx"][idx]) * dataset.step_size) + "\t")

                    writer.write(",".join([str(int(y)) for y in batch["labels"][idx].tolist() if y < pos_length and y > 0]) + "\t")
                    writer.write(",".join([str(int(y)) for y in outputs[idx].tolist() if y < pos_length]) + "\n")



if __name__ == '__main__':
    main()