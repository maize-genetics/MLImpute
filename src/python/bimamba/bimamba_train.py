import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
import wandb
from bimamba_train import BiMambaSmooth
import numba
import pandas as pd

with open("wandb_key.txt", 'r') as f:
    key = f.read().strip()

wandb.login(key=key)
scaler = torch.cuda.amp.GradScaler()


def gather_npy_paths(root_dir):
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(root_dir)
        for f in files if f.endswith(".npy")
    ]

class WindowIndexDataset(Dataset):
    def __init__(self, file_list, window_size=512, top_n=25, step_size=128, return_decode=False):
        self.entries = []
        self.window_size = window_size
        self.top_n = top_n
        self.step_size = step_size
        self.return_decode = return_decode
        for path in file_list:
            matrix = np.load(path, allow_pickle=True, mmap_mode='r')
            n_windows = (matrix.shape[0] - window_size) // step_size + 1
            self.entries.extend([(path, i) for i in range(n_windows)])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, window_idx = self.entries[idx]
        matrix = np.load(path, allow_pickle=True, mmap_mode='r')
        key = path.split("/")[2].split("_")[0]

        df = pd.read_csv(f"training_data/ps4g_weights/{key}.csv", sep='\t')
        weights = [None] * len(df)
        for _, row in df.iterrows():
            weights[row['gamete_index']] = row['weight']

        start = window_idx * self.step_size
        end = start + self.window_size
        window_matrix_unmasked = matrix[start:end]

        consecutive_hit = longest_consec(window_matrix_unmasked)
        parent_support = window_matrix_unmasked.sum(axis=0)
        combined = consecutive_hit + parent_support
        top_parents = np.argpartition(combined, -self.top_n)[-self.top_n:]
        top_parents = top_parents[np.argsort(combined[top_parents])[::-1]]

        weights = np.array(weights, dtype=np.float16)
        weight_vector = weights[top_parents]
        weighted_window = window_matrix_unmasked[:, top_parents] * weight_vector
        #unweighted_window = window_matrix_unmasked[:, top_parents]

        if self.return_decode:
            decode_info = top_parents.tolist()
            return (
                torch.tensor(weighted_window, dtype=torch.float32),
                torch.tensor(decode_info, dtype=torch.int64)
            )
        else:
            return torch.tensor(weighted_window, dtype=torch.float32)

# Evaluation Function
# def evaluate_model(model, test_loader, test_matrix, sequence_length, step_size, device, batch_size):
#     model.eval()
#     total_loss = 0.0
#     total_positions = test_matrix.shape[0]
#     label_counts = torch.zeros((total_positions, test_matrix.shape[1]), dtype=torch.int32, device=device)
#
#     with torch.no_grad():
#         for batch_idx, (batch_data, decode_dict) in enumerate(test_loader):
#             batch_data, decode_dict = batch_data.to(device), decode_dict.to(device)
#             outputs, mask = model(batch_data)
#             batch_predictions = torch.argmax(outputs, dim=-1)
#             loss = model.compute_loss(outputs, batch_data, mask)
#             total_loss += loss.item()
#             for i, pred in enumerate(batch_predictions):
#                 # get associated indices for window
#                 start = batch_idx * step_size * batch_size + i * step_size
#                 end = start + sequence_length
#                 for pos in range(sequence_length):
#                     if start + pos < total_positions:  # Avoid index overflow
#                         label = decode_dict[pred[pos]]
#                         label_counts[start + pos, label] += 1
#
#             if batch_idx % 1000 == 0:
#                 print(f"Validation Batch {batch_idx}/{len(test_loader)}, Loss: {loss.item():.4f}")
#
#     final_predictions = torch.argmax(label_counts, dim=-1)
#     avg_loss = total_loss / len(test_loader)
#     row_indices = torch.arange(total_positions, device=device)
#     snps = test_matrix[row_indices, final_predictions]
#     snp_accuracy = snps.float().mean().item()
#
#     wandb.log({"Loss": avg_loss, "Accuracy": snp_accuracy})
#
#     return avg_loss, snp_accuracy, final_predictions

def evaluate_model(model, test_loader, test_matrix, window_size, step_size, device, batch_size):
    model.eval()
    total_loss = 0.0
    total_positions, num_labels = test_matrix.shape
    label_counts = torch.zeros((total_positions, num_labels), dtype=torch.int32, device=device)

    with torch.no_grad():
        for batch_idx, (batch_data, decode_dict) in enumerate(test_loader):
            batch_data, decode_dict = batch_data.to(device), decode_dict.to(device)  # decode_dict: [B, top_n]
            outputs, mask = model(batch_data)
            batch_predictions = torch.argmax(outputs, dim=-1)  # [B, L]
            loss = model.compute_loss(outputs, batch_data, mask)
            total_loss += loss.item()

            B, L = batch_predictions.shape  # batch size and window size

            # Compute absolute genome positions for each predicted site
            start_indices = (
                batch_idx * step_size * batch_size + torch.arange(B, device=device) * step_size
            ).unsqueeze(1)  # [B, 1]
            pos_offsets = torch.arange(window_size, device=device).unsqueeze(0)  # [1, L]
            global_positions = (start_indices + pos_offsets).reshape(-1)  # [B*L]

            # Flatten predictions
            pred_labels = batch_predictions.reshape(-1)  # [B*L]

            # Construct per-row decode indexing
            row_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, L).reshape(-1)  # [B*L]
            label_ids = decode_dict[row_ids, pred_labels]  # [B*L]

            # Filter out-of-bounds positions
            valid_mask = global_positions < total_positions
            global_positions = global_positions[valid_mask]
            label_ids = label_ids[valid_mask]

            # Accumulate predicted label counts
            label_counts.index_put_(
                (global_positions, label_ids),
                torch.ones_like(label_ids, dtype=label_counts.dtype),
                accumulate=True
            )

            if batch_idx % 1000 == 0:
                print(f"Validation Batch {batch_idx}/{len(test_loader)}, Loss: {loss.item():.4f}")

    # Final SNP prediction via majority vote at each position
    final_predictions = torch.argmax(label_counts, dim=-1)
    avg_loss = total_loss / len(test_loader)
    snps = test_matrix[torch.arange(total_positions, device=device), final_predictions]
    snp_accuracy = snps.float().mean().item()

    wandb.log({"Loss": avg_loss, "Accuracy": snp_accuracy})
    return avg_loss, snp_accuracy, final_predictions



# Training Function
def train_model(model, train_loader, optimizer, epochs, device, test_loader, test_matrix, window_size, step_size,
                batch_size, save_path):
    wandb.init(project="BiMamba Imputation", name="bimamba-imputation", config={
        "epochs": epochs,
        "batch_size": train_loader.batch_size,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

    # Set intervals for mid-epoch checkpoints and evaluations
    checkpoint_interval = 1000
    eval_interval = 1000

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # ✅ Enable AMP
                outputs, mask = model(batch_data)
                loss = model.compute_loss(outputs, batch_data, mask)
            # ✅ Scale loss for AMP
            scaler.scale(loss).backward()

            # ✅ Unscale before gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ✅ Apply gradient clipping

            # ✅ Step optimizer using AMP
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
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
                    avg_val_loss, snp_accuracy, _ = evaluate_model(model, test_loader, test_matrix, window_size,
                                                            step_size, device, batch_size)
                    wandb.log({"Mid-Epoch Evaluation Loss": avg_val_loss,
                               "Accuracy": snp_accuracy,
                               "Step": epoch * len(train_loader) + batch_idx})
                    model.train()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_loss:.4f}")
        wandb.log({"Epoch Training Loss": avg_loss, "Epoch": epoch + 1})

        # ✅ Save end-of-epoch Model Checkpoint
        epoch_save_path = os.path.join(save_path, f"{epoch + 1}.pth")
        torch.save(model.state_dict(), epoch_save_path)
        wandb.save(epoch_save_path)

@numba.njit
def longest_consec(arr):
    n_rows, n_cols = arr.shape
    max_lengths = np.zeros(n_cols, dtype=np.int32)
    for col in range(n_cols):
        max_len = 0
        cur_len = 0
        for row in range(n_rows):
            if arr[row, col] == 1:
                cur_len += 1
                if cur_len > max_len:
                    max_len = cur_len
            else:
                cur_len = 0
        max_lengths[col] = max_len
    return max_lengths

def subset(matrix, window_size=512, top_n=25, exclude_index=None):
    step_size = window_size // 4
    num_rows, num_parents = matrix.shape
    # Manual slicing per window
    num_windows = (num_rows - window_size) // step_size + 1
    unmasked_windows = np.empty((num_windows, window_size, top_n), dtype=np.float32)
    decode_dict = []
    if exclude_index is not None:
        if isinstance(exclude_index, int):
            exclude_index = [exclude_index]
        exclude_index = set(exclude_index)
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        window_matrix_unmasked = matrix[start:end, :]
        # Compute scores
        consecutive_hit = longest_consec(window_matrix_unmasked)
        parent_support = window_matrix_unmasked.sum(axis=0)
        combined = consecutive_hit + parent_support
        if exclude_index:
            combined[list(exclude_index)] = -1
        top_parents = np.argpartition(combined, -top_n)[-top_n:]
        top_parents = top_parents[np.argsort(combined[top_parents])[::-1]]
        unmasked_windows[i] = window_matrix_unmasked[:, top_parents]
        decode_dict.append(top_parents.tolist())
    return unmasked_windows, decode_dict

def pad_arrays_columnwise(arrays, pad_value=0):
    max_cols = max(arr.shape[1] for arr in arrays)
    padded = []
    for arr in arrays:
        if arr.shape[1] < max_cols:
            pad_width = max_cols - arr.shape[1]
            padding = np.full((arr.shape[0], pad_width), pad_value, dtype=arr.dtype)
            arr_padded = np.hstack((arr, padding))
        else:
            arr_padded = arr
        padded.append(arr_padded)
    return padded

# ✅ Main Function
def main():
    wandb.init(project="BiMamba", name="bimamba_masked", entity="maize-genetics")

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

    args = parser.parse_args()

    window_size = args.window_size
    num_classes = args.num_classes
    batch_size = args.batch_size
    epochs = args.epochs
    d_model = args.d_model
    num_layers = args.num_layers
    step_size = window_size // 4
    lr = args.lr
    lambda_smooth = args.lambda_smooth

    print("Configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    model = BiMambaSmooth(input_dim=25, d_model=d_model, num_classes=num_classes, n_layer=num_layers, lambda_smooth=lambda_smooth, d_conv=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
                                  step_size=step_size, return_decode=True)

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
    train_model(model, train_loader, optimizer, epochs, device,
                test_loader, test_matrix, window_size, step_size, batch_size, save_path)
    evaluate_model(model, test_loader, test_matrix, window_size, step_size, device, batch_size)

    wandb.finish()


if __name__ == '__main__':
    main()