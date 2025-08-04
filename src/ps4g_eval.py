import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
from bimamba_model import BiMambaSmooth
from torch.utils.data import DataLoader, Dataset
import numba

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

        weights = np.load(f"training_data/weights/{key}_matrix_0.npz", allow_pickle=True)["weights"]
        
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

            if batch_idx % 100 == 0:
                print(f"Validation Batch {batch_idx}/{len(test_loader)}, Loss: {loss.item():.4f}")

    # Final SNP prediction via majority vote at each position
    final_predictions = torch.argmax(label_counts, dim=-1)
    avg_loss = total_loss / len(test_loader)
    snps = test_matrix[torch.arange(total_positions, device=device), final_predictions]
    snp_accuracy = snps.float().mean().item()

    return avg_loss, snp_accuracy, final_predictions

window_size = 512
num_classes = 25
batch_size = 64
d_model = 128
num_layers = 3
num_features = 25
step_size = window_size
lr = 1e-6
lambda_smooth = 0.2

model_checkpoint = "saved_models/weighted/5.pth"
model = BiMambaSmooth(input_dim=num_features, d_model=d_model, num_classes=num_classes, n_layer=num_layers)
print(model)

#print(torch.load(model_checkpoint))
model.load_state_dict(torch.load(model_checkpoint))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load in test matrix
test_paths = ["training_data/test/CML69_matrix.npy"]

test_dataset = WindowIndexDataset(test_paths, window_size=window_size, top_n=num_classes,
                                  step_size=step_size, return_decode=True)

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

avg_loss, snp_accuracy, predictions = evaluate_model(model, test_loader, test_matrix, window_size, step_size, device, batch_size)



ps4g_file = "../axial/ps4g_files/gvcfPS4GFiles_old/Zm-CML69-REFERENCE-NAM-1.0_fullASM_pos_matches2NM_ps4g.txt"
ps4g = pd.read_csv(ps4g_file, delimiter="\t", comment="#")

with open(ps4g_file, 'r') as file:
    comments = [line for line in file if line.startswith('#')]

ps4g['gameteSet'] = ps4g['gameteSet'].apply(lambda x: list(map(int, x.split(','))))
gamete_data = []

for line in comments:
    line = line.strip()
    if line.startswith("#") and ":" in line and "\t" in line:
        # Example line: "#B73:0\t1\t10730006"
        line = line[1:]  # Remove leading "#"
        gamete_full, idx, count = line.split("\t")
        gamete_name = gamete_full.split(":")[0]
        gamete_data.append({
            "gamete": gamete_name,
            "gamete_index": int(idx),
        })

# collapse positions
collapsed_df = ps4g.groupby('pos').agg({
    'gameteSet': lambda x: list(set().union(*x)),  # union all sets
    'count': 'sum'  # sum counts
}).reset_index()

ps4g_preds = ps4g[:len(predictions)]
index_to_name = {entry["gamete_index"]: entry["gamete"] for entry in gamete_data}
max_index = max(index_to_name.keys())  # ensure all indices fit
index_array = [index_to_name[i] for i in range(max_index + 1)]
predicted_parents = np.array(index_array)[predictions.cpu().numpy()]

ps4g_preds['parents'] = predicted_parents
agg = ps4g_preds.groupby(['pos', 'parents']).size().unstack(fill_value=0)
print("aggregated CML69: ", (agg['CML69'] != 0).sum()/len(agg))

# Print results
most_common_parents = agg.idxmax(axis=1)
proportion_cml69_top = (most_common_parents == 'CML69').sum() / len(most_common_parents)

print("Proportion where CML69 is the most common prediction: ", proportion_cml69_top)

filtered_predictions = np.array(most_common_parents)

values, counts = np.unique(filtered_predictions, return_counts=True)

for val, count in zip(values, counts):
    percentage = count / len(filtered_predictions) * 100
    print(f"{val}: {percentage:.2f}%")
