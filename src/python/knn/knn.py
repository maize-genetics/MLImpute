import numpy as np
import logging
from tqdm import tqdm

def run_knn(matrix, window_size, diploid, homozygote_threshold=0.9):
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
    assert window_size % 2 == 1, "Window size must be odd to have a center position."
    num_positions, num_samples = matrix.shape
    half_window = window_size // 2

    path1 = []
    path2 = []

    # Predict each site in the full matrix window
    for center in tqdm(range(num_positions)):
        start = max(center - half_window, 0)
        end = min(center + half_window + 1, num_positions)

        window = matrix[start:end, :]
        sample_sums = window.sum(axis=0)  # shape: (num_samples,)

        # Find indices of top two parents (highest counts)
        top_parents = np.argsort(sample_sums)[-2:][::-1]
        p1, p2 = top_parents
        s1, s2 = sample_sums[p1], sample_sums[p2]


        # Homozygote vs heterozygote call
        denom = (s1 + s2) if (s1 + s2) > 0 else 1.0
        top1_share = s1 / denom
        top2_share = s2 / denom
        diff = top1_share - top2_share
        if not diploid:
            # If not diploid, always return the top parent
            path1.append(p1)
            path2.append(p1)
        elif diff >= homozygote_threshold:
            path1.append(p1)
            path2.append(p1)
        else:
            path1.append(p1)
            path2.append(p2)



    # Return the path of predicted samples
    return np.stack([path1, path2], axis=1)
