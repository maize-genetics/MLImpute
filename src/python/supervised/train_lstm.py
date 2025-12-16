import argparse
import torch
from torch.utils.data import DataLoader
import os
from src.python.supervised.deprecated.models_labeled import LstmSegmentationEncDec
from src.python.supervised.deprecated.dataset_labeled import BaseSegmentationDataset
import wandb
from tqdm import tqdm
import sys


def train(model, train_loader, test_loader, optimizer, criterion, epochs, save_path,
          device, checkpoint_interval=5000, eval_interval=5000, report_interval=100):
    wandb.init(project="LSTM Encoder-Decoder Segmentation", name="lstm-segmentation", config={
        "epochs": epochs,
        "batch_size": train_loader.batch_size,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

    # Set intervals for mid-epoch checkpoints and evaluations
    checkpoint_interval = 5000
    eval_interval = 5000

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        print("training epoch " + str(epoch))
        for batch_idx, batch in tqdm(enumerate(train_loader), desc="Training..."):
            optimizer.zero_grad()

            input_ids = batch["encoder_hidden_states"].to(device)
            labels = batch["labels"].to(dtype=torch.float32, device=device)
            attention_mask = batch["attention_mask"].to(device)

            output = model(input_ids, labels)
            loss = torch.sum(criterion(output, labels.unsqueeze(2), reduction="none") * attention_mask.unsqueeze(2))
            loss = loss / torch.sum(attention_mask)  # mean loss
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            wandb.log({"Training Loss": loss.item()})

            if batch_idx % 1000 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

            # Save a mid-epoch checkpoint every checkpoint_interval batches
            if (batch_idx + 1) % checkpoint_interval == 0:
                mid_epoch_save_path = os.path.join(save_path, f"epoch_{epoch + 1}_batch_{batch_idx + 1}.pth")
                torch.save(model.state_dict(), mid_epoch_save_path)
                wandb.save(mid_epoch_save_path)
                print(f"Saved mid-epoch checkpoint at Epoch {epoch + 1}, Batch {batch_idx + 1}")

                # Evaluate the model every eval_interval batches
            if (batch_idx + 1) % eval_interval == 0:
                print(f"Evaluating model at Epoch {epoch + 1}, Batch {batch_idx + 1}")
                model.eval()
                avg_val_loss = evaluate(model, test_loader, criterion, device)
                wandb.log({"Mid-Epoch Evaluation Loss": avg_val_loss.item()})
                model.train()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_loss:.4f}")
        wandb.log({"Epoch Training Loss": avg_loss, "Epoch": epoch + 1})

        # ✅ Save end-of-epoch Model Checkpoint
        epoch_save_path = os.path.join(save_path, f"{epoch + 1}.pth")
        torch.save(model.state_dict(), epoch_save_path)
        wandb.save(epoch_save_path)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating..."):
            input_ids = batch_data["encoder_hidden_states"].to(device)
            labels = batch_data["labels"].to(dtype=torch.float32, device=device)
            attention_mask = batch_data["attention_mask"].to(device)

            output = model(input_ids, labels)
            loss = torch.sum(criterion(output, labels) * attention_mask)

            epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    model.train()
    return avg_loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyfile", type=str, required=True, help="keyfile with list of input files")
    parser.add_argument("--windows", type=str, default=None, help="Optional, specify which windows to include")
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--num-parents", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--learning-rateq", type=float, default=1e-4)
    parser.add_argument("--save-model-path", type=str, default="saved_models/")
    parser.add_argument("--step-size", type=int, default=128)
    parser.add_argument("--allow-cpu", action="store_true", help="allow cpu for training (not recommended)")

    args = parser.parse_args()

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

    window_size = args.window_size
    num_parents = args.num_parents
    batch_size = args.batch_size
    epochs = args.epochs
    d_model = args.d_model
    num_layers = args.num_layers
    step_size = args.step_size
    lr = args.lr

    print("Configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    model = LstmSegmentationEncDec(num_parents, d_model, num_layers, 0.1)

    criterion = torch.nn.functional.mse_loss

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    save_path = args.save_model_path
    os.makedirs(save_path, exist_ok=True)

    dataset = BaseSegmentationDataset(args.keyfile, windows=args.windows, input_len=window_size, step_size=step_size, end_token=1000)

    dataset_chunks = torch.utils.data.random_split(dataset, [0.95, 0.05])
    dataset_train = dataset_chunks[0]
    dataset_val = dataset_chunks[1]

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=dataset_train.collate)
    test_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn = dataset_val.collate)

    # Pass the evaluation arguments to train_model
    train(model, train_loader, test_loader, optimizer, criterion, epochs, save_path, device)

    wandb.finish()


if __name__ == '__main__':
    main()