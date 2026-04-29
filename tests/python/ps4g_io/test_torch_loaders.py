import numpy as np
import torch
import pytest

from python.ps4g_io.torch_loaders import SimpleDataset, build_simple_dataloader

def test_simple_dataset():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    dataset = SimpleDataset(data)

    assert len(dataset) == 3
    assert torch.equal(dataset[0], torch.tensor([1.0, 2.0]))
    assert torch.equal(dataset[2], torch.tensor([5.0, 6.0]))

def test_build_simple_dataloader():
    data = np.array([[10, 20], [30, 40], [50, 60], [70, 80]])
    loader = build_simple_dataloader(data, batch_size=2, shuffle=False)

    batches = list(loader)
    assert len(batches) == 2  # 4 samples / batch size 2
    assert batches[0].shape == (2, 2)
    assert torch.equal(batches[0], torch.tensor([[10.0, 20.0], [30.0, 40.0]]))

# -----------------------
# Corner Case Tests Below
# -----------------------

def test_empty_dataset():
    data = np.empty((0, 2))
    dataset = SimpleDataset(data)
    assert len(dataset) == 0
    with pytest.raises(IndexError):
        _ = dataset[0]

def test_empty_dataloader():
    data = np.empty((0, 2))
    loader = build_simple_dataloader(data, batch_size=2, shuffle=False)
    assert len(list(loader)) == 0

def test_single_sample_dataset():
    data = np.array([[42, 99]])
    loader = build_simple_dataloader(data, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 1
    assert batches[0].shape == (1, 2)
    assert torch.equal(batches[0], torch.tensor([[42.0, 99.0]]))

def test_non_numeric_input():
    data = np.array([["a", "b"], ["c", "d"]])
    with pytest.raises(ValueError):
        _ = build_simple_dataloader(data)