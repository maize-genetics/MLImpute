from seq2seq_diploid import LabeledDatasetDiploid
import argparse
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from seq2seq_diploid import Seq2SeqDiploid, EncoderDiploid, DecoderDiploid

def inference(model, iterator):
    device=model.device
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating..."):
            input_embeds = batch["input_embeds"].to(device)
            batch_predictions = model(input_embeds)
            pred_y = torch.argmax(batch_predictions, dim=-1).detach().cpu().to(torch.int64)
            predictions.append(pred_y)
    # concat batches along batch dimension -> [N, T]
    preds = torch.cat(predictions, dim=0)  # shape consistent even if last batch smaller
    return preds.numpy()


# arguments for running the script
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, required=True, help="path to the input data")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="batch size")
    parser.add_argument("--model-path", type=str, default="best_model.pt", help="path to model")
    parser.add_argument("--save-dir", type=str, required=True, help="path to save imputed paths")
    parser.add_argument("--num-parents", "--np", type=int, default=24, help="number of parents")
    parser.add_argument("--max-seq-length", "--sl", type=int, default=512, help="maximum input sequence length")
    parser.add_argument("--step-size", "-s", type=int, default=128, help="distance between the start points of each training window")
    parser.add_argument("--embedding-dim", type=int, default=12, help="embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=24, help="hidden dimension")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initializing model
    ckpt = torch.load(args.model_path, map_location=device)
    EMB_DIM = args.embedding_dim
    HID_DIM = args.hidden_dim

    enc = EncoderDiploid(args.num_parents, EMB_DIM, HID_DIM)
    dec = DecoderDiploid(args.num_parents+1, EMB_DIM, HID_DIM, ploidy=2)
    model = Seq2SeqDiploid(enc, dec, device, ploidy=2)
    model.load_state_dict(ckpt)
    model.to(device)

    os.makedirs(args.save_dir, exist_ok=True)

    filenames = os.listdir(args.data_path)
    for file in filenames:
        dataset = LabeledDatasetDiploid(args.data_path, [file], args.max_seq_length, args.num_parents, args.step_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        predictions = inference(model, dataloader)
        pred_file = file.split("_matrix")[0]
        np.save(os.path.join(args.save_dir, pred_file), predictions)


if __name__ == "__main__":
    main()