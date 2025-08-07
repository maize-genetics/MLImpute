import numba
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SimpleDataset(Dataset):
    """
        A simple dataset class for loading numeric numpy arrays into PyTorch tensors.

        This class takes a numeric numpy array as input and provides a PyTorch Dataset
        interface, returning each item as a torch.float32 tensor. Useful for quickly
        wrapping numpy data for use with PyTorch DataLoader.
    """
    def __init__(self, X):
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("Input data must be numeric.")
        self.X = X.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32)

#TODO determine if we need to set batch_size(likely) and shuffle (likely not) as parameters
def build_simple_dataloader(X, batch_size=512, shuffle=True):
    """
    Build a DataLoader for the dataset.

    Args:
        X (np.ndarray): Input data.
        batch_size (int): Size of each batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    dataset = SimpleDataset(X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)


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

        start = window_idx * self.step_size
        end = start + self.window_size
        window_matrix_unmasked = matrix[start:end]

        consecutive_hit = longest_consec(window_matrix_unmasked)
        parent_support = window_matrix_unmasked.sum(axis=0)
        combined = consecutive_hit + parent_support
        top_parents = np.argpartition(combined, -self.top_n)[-self.top_n:]
        top_parents = top_parents[np.argsort(combined[top_parents])[::-1]]

        weights = self.build_weights(key)
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

    def build_weights(self, key):
        df = pd.read_csv(f"training_data/ps4g_weights/{key}.csv", sep='\t')
        weights = [None] * len(df)
        for _, row in df.iterrows():
            weights[row['gamete_index']] = row['weight']
        return np.array(weights, dtype=np.float16)


class WindowIndexDatasetFromMatrix(Dataset):
    def __init__(self, matrix,weights, window_size=512, top_n=25, step_size=128, return_decode=False):
        self.matrix = matrix
        self.weights = weights
        self.window_size = window_size
        self.top_n = top_n
        self.step_size = step_size
        self.return_decode = return_decode
        self.n_windows = (matrix.shape[0] - window_size) // step_size + 1

    def __len__(self):
        return len(self.n_windows)

    def __getitem__(self, idx):
        start = idx * self.step_size # idx is the window index now as we are only loading from one matrix
        end = start + self.window_size
        window_matrix_unmasked = self.matrix[start:end]

        consecutive_hit = longest_consec(window_matrix_unmasked)
        parent_support = window_matrix_unmasked.sum(axis=0)
        combined = consecutive_hit + parent_support
        top_parents = np.argpartition(combined, -self.top_n)[-self.top_n:]
        top_parents = top_parents[np.argsort(combined[top_parents])[::-1]]

        weight_vector = self.weights[top_parents]
        weighted_window = window_matrix_unmasked[:, top_parents] * weight_vector

        if self.return_decode:
            decode_info = top_parents.tolist()
            return (
                torch.tensor(weighted_window, dtype=torch.float32),
                torch.tensor(decode_info, dtype=torch.int64)
            )
        else:
            return torch.tensor(weighted_window, dtype=torch.float32)

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
