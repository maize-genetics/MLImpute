import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
from bimamba_model import BiMambaSmooth
from torch.utils.data import DataLoader, Dataset
import numba
from bimamba_model import BiMambaSmooth
from bimamba_train import WindowIndexDataset

def decode_position(encoded_pos):
    """
    Decode a 32-bit integer that packs:
      • the upper-8 bits → an index (0-255)
      • the lower-24 bits → a position, but quantised in bins of 256 bp
    This is lossy because we multiplied by 256 during encoding.
    """
    idx = (encoded_pos >> 24) & 0xFF           # top-byte index (unsigned)
    pos = (encoded_pos & 0x0FFFFFF) * 256      # restore to bp units
    return idx, pos


def main():
    window_size = 512
    num_classes = 25
    batch_size = 64
    d_model = 128
    num_layers = 3
    num_features = 25
    step_size = window_size
    lambda_smooth = 0.2

    model = BiMambaSmooth(input_dim=num_features, d_model=d_model, num_classes=num_classes, n_layer=num_layers, lambda_smooth=lambda_smooth)
    model_checkpoint = "saved_models/weighted/5.pth"
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

    # TODO: run eval loop and get two predictions
    # TODO: get hidden states and connect to HMM