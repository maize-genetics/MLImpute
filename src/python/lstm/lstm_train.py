import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import wandb

from encoder_decoder import Encoder, Decoder, Seq2Seq, WindowIndexDataset


def gather_npy_paths(root_dir):
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(root_dir)
        for f in files if f.endswith(".npy")
    ]


def train(model, train_loader, test_loader, optimizer, criterion, epochs, save_path):
    wandb.init(project="LSTM Encoder-Decoder Imputation", name="lstm-imputation", config={
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
        for batch_idx, batch_data in enumerate(train_loader):

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                B, L, N = batch_data.shape
                batch_data = batch_data.to(model.device)
                mask = torch.rand(B, L, device=batch_data.device) < 0.15  # randomly change 15% of input to 0 for training
                input_masked = batch_data.masked_fill(mask.unsqueeze(-1), 0)
                #input_masked = input_masked.to(model.device)

                output = model(input_masked)
                loss = criterion(output, batch_data.masked_fill(batch_data > 0, 1), mask.unsqueeze(2).repeat(1, 1, N))
                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()

                wandb.log({"Training Loss": loss.item(), "Step": epoch * len(train_loader) + batch_idx})

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
                        avg_val_loss = evaluate(model, test_loader, criterion)
                        wandb.log({"Mid-Epoch Evaluation Loss": avg_val_loss,
                                   "Step": epoch * len(train_loader) + batch_idx})
                        model.train()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_loss:.4f}")
        wandb.log({"Epoch Training Loss": avg_loss, "Epoch": epoch + 1})

        # ✅ Save end-of-epoch Model Checkpoint
        epoch_save_path = os.path.join(save_path, f"{epoch + 1}.pth")
        torch.save(model.state_dict(), epoch_save_path)
        wandb.save(epoch_save_path)


def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = batch_data.to(model.device)
            output = model(batch_data)
            loss = criterion(output, batch_data)
            epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    wandb.log({"Loss": avg_loss})
    model.train()
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_smooth", type=float, default=0.2)
    parser.add_argument("--save_path", type=str, default="saved_models/")
    parser.add_argument("--step_size", type=int, default=512)

    args = parser.parse_args()

    window_size = args.window_size
    num_classes = args.num_classes
    batch_size = args.batch_size
    epochs = args.epochs
    d_model = args.d_model
    num_layers = args.num_layers
    step_size = args.step_size
    lr = args.lr
    lambda_smooth = args.lambda_smooth

    print("Configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, device)
    # dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    # model = Seq2Seq(enc, dec, device).to(device)

    encoder = Encoder(emb_dim=25, hid_dim=512, n_layers=3, dropout=0.5, device=device)
    decoder = Decoder(output_dim=25, emb_dim=512, hid_dim=512, n_layers=6, dropout=0.5)
    model = Seq2Seq(encoder=encoder, decoder=decoder, device=device)
    #model = BiMambaSmooth(input_dim=25, d_model=d_model, num_classes=num_classes, n_layer=num_layers, lambda_smooth=lambda_smooth, d_conv=4)
    #criterion = torch.nn.BCELoss(reduction="mean")
    criterion = torch.nn.functional.binary_cross_entropy_with_logits

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(model)

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    train_paths = gather_npy_paths("training_data/train")
    test_paths = gather_npy_paths("training_data/test")

    train_dataset = WindowIndexDataset(train_paths, window_size=window_size, top_n=num_classes,
                                   step_size=step_size, return_decode=False)
    test_dataset = WindowIndexDataset(test_paths, window_size=window_size, top_n=num_classes,
                                  step_size=window_size, return_decode=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    # Reconstruct test_matrix just for SNP accuracy computation
    test_matrix_parts = []
    for path in test_paths:
        matrix = np.load(path, allow_pickle=True, mmap_mode='r')
        end = matrix.shape[0] - (matrix.shape[0] % window_size)
        truncated_matrix = matrix[:end]
        test_matrix_parts.append(truncated_matrix)

    test_matrix = np.concatenate(test_matrix_parts, axis=0)
    test_matrix = torch.tensor(test_matrix, dtype=torch.float32, device=device)

    # Pass the evaluation arguments to train_model
    train(model, train_loader, test_loader, optimizer, criterion, epochs, save_path)
    evaluate(model, test_loader, test_matrix, criterion)

    wandb.finish()


if __name__ == '__main__':
    main()