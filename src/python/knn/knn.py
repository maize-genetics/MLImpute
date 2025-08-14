import numpy as np
import logging
from tqdm import tqdm

def run_knn(matrix, window_size, diploid, pseudocount=1e-6, homozygote_threshold=0.9):
    """
    Uses a sliding window to predict the most likely sample for each center position,
    then checks if that sample shares the allele at the center position.

    Args:
        matrix (np.ndarray): A 2D array of shape (positions, samples) with values 0 or 1.
        window_size (int): Size of the sliding window (should be odd to have a center).
        diploid (bool): If True, returns a 2 paths of predicted samples for each center position.
                        If False, returns a single path predicted sample for each center position.

    Returns:
        path (np.ndarray): Path of predicted samples for each center position.
    """
    window_size = window_size #window size is k
    assert window_size % 2 == 1, "Window size must be odd to have a center position."
    num_positions, num_samples = matrix.shape
    half_window = window_size // 2

    path1 = []
    path2 = []


    # Normalize counts to proportions per site
    counts = matrix + pseudocount
    props = counts / counts.sum(axis=1, keepdims=True)  # shape: (positions, parents)

    # Predict each site in the full matrix window
    for center in tqdm(range(half_window, num_positions - half_window)):
        start = max(center - half_window,0)
        end = min(center + half_window + 1, num_positions)

        # Take local window proportions and average across window
        window_props = props[start:end, :]
        mean_props = window_props.mean(axis=0)

        # Identify top-2 parents
        top_parents = np.argsort(mean_props)[-2:][::-1]  # sorted descending
        top_scores = mean_props[top_parents]

        # Decide homozygote vs heterozygote
        if not diploid or top_scores[0] >= homozygote_threshold:
            path1.append(top_parents[0])
            path2.append(top_parents[0])
        else:
            path1.append(top_parents[0])
            path2.append(top_parents[1])

        # window = matrix[start:end, :]
        # sample_sums = window.sum(axis=0)
        # predicted_sample = np.argmax(sample_sums)
        #
        # path1.append(predicted_sample)
        # path2.append(predicted_sample)


    # Return the path of predicted samples
    return np.stack([path1, path2], axis=1)
