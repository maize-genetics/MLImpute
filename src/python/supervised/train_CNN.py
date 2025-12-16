import argparse
import torch
from torch.utils.data import DataLoader
import os
from src.python.supervised.deprecated.dataset_labeled import CategoricalDataset, CategoricalMulticlassDataset
import wandb
from tqdm import tqdm
import sys
from src.python.supervised.deprecated.models_labeled import CNN, CNN2D, FullyConnected


def train(model, train_loader, test_loader, optimizer, criterion, epochs, save_path,
          device, checkpoint_interval=5000, eval_interval=5000, report_interval=100, project_name="train_CNN", run_name="cnn-test"):
    wandb.init(project=project_name, name=run_name, config={
        "epochs": epochs,
        "batch_size": train_loader.batch_size,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        print("training epoch " + str(epoch))
        for batch_idx, batch in tqdm(enumerate(train_loader), desc="Training..."):
            optimizer.zero_grad()

            input_ids = batch["inputs_embeds"].to(device)
            labels = batch["labels"].to(device=device) # TODO: data type handling

            output = model(input_ids)
            loss = criterion(output.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % report_interval == 0:
                wandb.log({"Training Loss": loss.item()})
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
                wandb.log({"Mid-Epoch Evaluation Loss": avg_val_loss})
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
            input_ids = batch_data["inputs_embeds"].to(device)
            labels = batch_data["labels"].to(dtype=torch.float32, device=device)

            output = model(input_ids)
            loss = criterion(output.squeeze(), labels.squeeze())

            epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    model.train()
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyfile", type=str, required=True, help="keyfile with list of input files")
    parser.add_argument("--windows", type=str, default=None, help="Optional, specify which windows to include")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional, checkpoint to load")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-model-path", type=str, default="saved_models/")
    parser.add_argument("--step-size", type=int, default=8)
    parser.add_argument("--mode", type=str, default="binary", help="training mode: binary or multiclass")
    parser.add_argument("--allow-cpu", action="store_true", help="allow cpu for training (not recommended)")
    parser.add_argument("--project-name", type=str, default="train-CNN", help="wandb project name")
    parser.add_argument("--run-name", type=str, default="cnn-test", help="wandb run name")
    parser.add_argument("--conv", type=str, default="1d", help="type of convolutions: 1d, 2d, or none")
    parser.add_argument("--padding", type=int, default=0, help="context padding")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        if args.allow_cpu:
            device = torch.device('cpu')
        else:
            print("Error: GPU not found")
            sys.exit()

    batch_size = args.batch_size
    epochs = args.num_epochs
    step_size = args.step_size
    lr = args.learning_rate

    print("Configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    if args.mode == "binary":
        if args.conv == "2d" or args.conv == "2D":
            model = CNN2D()
        elif args.conv == "1d" or args.conv == "1D":
            model = CNN()
        else:
            model = FullyConnected()
        criterion = torch.nn.functional.binary_cross_entropy
        dataset = CategoricalDataset(args.keyfile, windows=args.windows, input_len=32, step_size=step_size,
                                     preload=True, padding=args.padding)


    else:
        if args.conv == "2d" or args.conv == "2D":
            model = CNN(output_dim=17, sigmoid=False)
        elif args.conv == "1d" or args.conv == "1D":
            model = CNN2D(output_dim=17, sigmoid=False)
        else:
            model = FullyConnected(output_dim=17, sigmoid=False)
        loss_weights = [1] * 17
        loss_weights[16] = 0.0624
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(loss_weights).to(device))
        dataset = CategoricalMulticlassDataset(args.keyfile, windows=args.windows, input_len=32, step_size=step_size,
                                               preload=True)


    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    save_path = args.save_model_path
    os.makedirs(save_path, exist_ok=True)

    dataset_chunks = torch.utils.data.random_split(dataset, [0.95, 0.05])
    dataset_train = dataset_chunks[0]
    dataset_val = dataset_chunks[1]

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=16)
    test_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=16)

    # Pass the evaluation arguments to train_model
    train(model, train_loader, test_loader, optimizer, criterion, epochs, save_path, device, project_name=args.project_name,
          run_name=args.run_name)

    wandb.finish()


if __name__ == '__main__':
    main()