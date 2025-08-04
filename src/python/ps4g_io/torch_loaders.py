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