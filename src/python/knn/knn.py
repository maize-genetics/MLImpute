import numpy as np
import logging
from tqdm import tqdm

def run_knn(args, matrix):
    """
    Uses a sliding window to predict the most likely sample for each center position,
    then checks if that sample shares the allele at the center position.

    Args:
        matrix (np.ndarray): A 2D array of shape (positions, samples) with values 0 or 1.
        window_size (int): Size of the sliding window (should be odd to have a center).

    Returns:
        path (np.ndarray): Path of predicted samples for each center position.
    """
    window_size = args.window_size #window size is k
    assert window_size % 2 == 1, "Window size must be odd to have a center position."
    num_positions, num_samples = matrix.shape
    logging.info("Num positions: {}".format(num_positions))
    half_window = window_size // 2

    path = []

    # Predict each site in the full matrix window

    for center in tqdm(range(half_window, num_positions - half_window)):
        start = max(center - half_window,0)
        end = min(center + half_window + 1, num_positions)

        window = matrix[start:end, :]
        sample_sums = window.sum(axis=0)
        predicted_sample = np.argmax(sample_sums)

        path.append(predicted_sample)


    # Return the path of predicted samples
    return path