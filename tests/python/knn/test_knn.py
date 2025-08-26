# File: tests/test_knn.py

import numpy as np
import pytest
from python.knn.knn import run_knn

def test_run_knn_basic_haploid():
    matrix = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    result = run_knn(matrix, window_size=3, diploid=False)
    # Should predict the dominant sample in each window
    assert result.shape == (5, 2)
    assert np.all(result[:, 0] == result[:, 1])
    assert result[0,0] == 0  # First two positions dominated by sample 0
    assert result[1,0] == 0
    assert result[2,0] == 1  # Middle two positions dominated by sample 1
    assert result[3,0] == 1
    assert result[4,0] == 2  # Last position dominated by

def test_run_knn_basic_diploid():
    matrix = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    result = run_knn(matrix, window_size=3, diploid=True)
    assert result.shape == (5, 2)
    # Should return two different samples for heterozygous windows
    assert np.any(result[:, 0] != result[:, 1])
    assert result[0,0] == 0 and result[0,1] == 0 # First two positions dominated by sample 0
    assert (result[1,0] == 0 and result[1,1] == 1) or (result[1,0] == 1 and result[1,1] == 0)
    assert (result[2,0] == 0 and result[2,1] == 1) or (result[2,0] == 1 and result[2,1] == 0)
    assert (result[3,0] == 1 and result[3,1] == 2) or (result[3,0] == 2 and result[3,1] == 1)
    assert (result[4,0] == 1 and result[4,1] == 2) or (result[4,0] == 2 and result[4,1] == 1)


def test_run_knn_homozygote_threshold():
    matrix = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ])
    # # Set a high threshold so only perfect matches are called homozygote
    result = run_knn(matrix, window_size=3, diploid=True, homozygote_threshold=0.99)
    assert result.shape == (5, 2)
    assert np.any(result[:, 0] != result[:, 1])
    assert (result[0, 0] == 0 and result[0, 1] == 1) or (result[0, 0] == 1 and result[0, 1] == 0)
    assert (result[1, 0] == 0 and result[1, 1] == 1) or (result[1, 0] == 1 and result[1, 1] == 0)
    assert (result[2, 0] == 1 and result[2, 1] == 2) or (result[2, 0] == 2 and result[2, 1] == 1)
    assert (result[3, 0] == 1 and result[3, 1] == 2) or (result[3, 0] == 2 and result[3, 1] == 1)
    assert (result[4, 0] == 1 and result[4, 1] == 2) or (result[4, 0] == 2 and result[4, 1] == 1)

    result = run_knn(matrix, window_size=3, diploid=True, homozygote_threshold=0.49)
    assert result.shape == (5, 2)
    print(result)
    assert np.any(result[:, 0] != result[:, 1])
    assert (result[0, 0] == 0 and result[0, 1] == 1) or (result[0, 0] == 1 and result[0, 1] == 0)
    assert (result[1, 0] == 0 and result[1, 1] == 1) or (result[1, 0] == 1 and result[1, 1] == 0)
    assert (result[2, 0] == 1 and result[2, 1] == 1)
    assert (result[3, 0] == 1 and result[3, 1] == 2) or (result[3, 0] == 2 and result[3, 1] == 1)
    assert (result[4, 0] == 1 and result[4, 1] == 2) or (result[4, 0] == 2 and result[4, 1] == 1)

