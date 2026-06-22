from src.python.supervised.train_supervised import BertTagger, LabeledDataset
import argparse
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from transformers import ModernBertModel, ModernBertConfig


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
    parser.add_argument("--num-hidden-layers", "--nh", type=int, default=12, help="number of hidden layers in BERT")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initializing model
    ckpt = torch.load(args.model_path, map_location=device)
    configuration = ModernBertConfig(num_hidden_layers=args.num_hidden_layers, max_position_embeddings=args.max_seq_length)
    model = BertTagger(ModernBertModel(configuration), args.num_parents, 0.0, device=device)
    model.load_state_dict(ckpt)
    model.to(device)

    os.makedirs(args.save_dir, exist_ok=True)

    filenames = os.listdir(args.data_path)
    for file in filenames:
        dataset = LabeledDataset(args.data_path, [file], args.max_seq_length, args.num_parents, args.step_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        predictions = inference(model, dataloader)
        pred_file = file.split("_matrix")[0]
        np.save(os.path.join(args.save_dir, pred_file), predictions)
        #np.save(os.path.join(args.save_dir, file), predictions_bestpt)



if __name__ == "__main__":
    main()