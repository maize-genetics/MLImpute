from tqdm import tqdm
import torch
import argparse
import sys
from src.python.supervised.deprecated.models_labeled import CNN, CNN2D
from src.python.supervised.deprecated.dataset_labeled import CategoricalDataset
from torch.utils.data import DataLoader

# arguments for running the script
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True, help="path to a previous training checkpoint")
    parser.add_argument("--keyfile", type=str, required=True, help="keyfile with list of input files")
    parser.add_argument("--allow-cpu",  action="store_true", help="allow cpu for training (not recommended)")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="batch size")
    parser.add_argument("--output", type=str, required=True, help="path to output file")
    parser.add_argument("--conv-2d", action="store_true", help="if true, use 2-dimensional convolution")
    parser.add_argument("--padding", type=int, default=0, help="context padding")
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


    if args.conv_2d:
        model = CNN2D()
    else:
        model = CNN()


    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict)

    dataset = CategoricalDataset(args.keyfile, input_len=32, step_size=args.step_size,
                                 preload=True, padding=args.padding, include_index=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # set up model
    model = model.to(device)
    model.eval()

    with open(args.output, "w") as writer:
        writer.write("file_idx\tstart_pos\tactual\tpredicted\n")


        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating..."):
                input_ids = batch["inputs_embeds"].to(device)

                outputs = model(input_ids).cpu().detach().squeeze()

                for idx in range(args.batch_size):
                    writer.write(str(int(batch["file_idx"][idx])) + "\t" + str(int(batch["pos_idx"][idx]) * dataset.step_size) + "\t")
                    writer.write(str(int(batch["labels"][idx])) + "\t")
                    writer.write(str(float(outputs[idx])) + "\n")


if __name__ == '__main__':
    main()
