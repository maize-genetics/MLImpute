#!/usr/bin/env python3
"""
Extract windows from numpy arrays in nested directories and save as one large file.
Handles arrays of shape (time_steps, numParents) and extracts windows of shape (window_size, numParents).
"""

import numpy as np
import os
import argparse
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
import random
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_windows_2d(data: np.ndarray, window_size: int = 512, step_size: int = 512) -> np.ndarray:
    """
    Extract windows from 2D array while preserving the second dimension.

    Args:
        data: Input numpy array of shape (time_steps, features) or (time_steps,)
        window_size: Size of each window along the first dimension
        step_size: Step between windows along the first dimension

    Returns:
        Array of windows with shape (n_windows, window_size, features) or (n_windows, window_size)
    """
    if data.ndim == 1:
        # Handle 1D arrays
        if len(data) < window_size:
            return np.array([]).reshape(0, window_size)

        windows = sliding_window_view(data, window_shape=window_size)
        return windows[::step_size]

    elif data.ndim == 2:
        # Handle 2D arrays - extract windows along first dimension
        time_steps, num_features = data.shape

        if time_steps < window_size:
            return np.array([]).reshape(0, window_size, num_features)

        # Use sliding_window_view along the first axis only
        windows = sliding_window_view(data, window_shape=window_size, axis=0)

        # Select every step_size-th window
        selected_windows = windows[::step_size]

        # Reshape to (n_windows, window_size, num_features)
        n_windows = selected_windows.shape[0]
        return selected_windows.reshape(n_windows, window_size, num_features)

    else:
        # For 3D+ arrays, flatten to 1D
        logger.warning(f"Array has {data.ndim} dimensions. Flattening to 1D.")
        flattened = data.flatten()
        return extract_windows_2d(flattened, window_size, step_size)


def get_array_info(data: np.ndarray) -> str:
    """Get readable info about array shape and size."""
    size_mb = data.nbytes / (1024 * 1024)
    return f"shape={data.shape}, dtype={data.dtype}, size={size_mb:.2f}MB"


def find_numpy_files(root_directory: Path) -> List[Path]:
    """
    Recursively find all .npy files in directory and subdirectories.

    Args:
        root_directory: Root directory to search

    Returns:
        List of paths to .npy files
    """
    npy_files = []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(root_directory):
        root_path = Path(root)
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(root_path / file)

    return sorted(npy_files)


def estimate_total_windows(npy_files: List[Path], window_size: int, step_size: int,
                           sample_size: int = 10) -> tuple:
    """
    Estimate the total number of windows and memory requirements by sampling files.

    Args:
        npy_files: List of numpy file paths
        window_size: Size of each window
        step_size: Step between windows
        sample_size: Number of files to sample for estimation

    Returns:
        Tuple of (estimated_total_windows, estimated_memory_gb, detected_features, detected_dtype)
    """
    logger.info("Estimating total windows and memory requirements...")

    sample_files = npy_files[:min(sample_size, len(npy_files))]
    total_windows_sampled = 0
    detected_features = None
    detected_dtype = None

    for npy_file in sample_files:
        try:
            data = np.load(npy_file)
            if detected_dtype is None:
                detected_dtype = data.dtype

            if data.ndim == 2 and detected_features is None:
                detected_features = data.shape[1]

            windows = extract_windows_2d(data, window_size, step_size)
            total_windows_sampled += len(windows)

        except Exception as e:
            logger.warning(f"Could not sample {npy_file}: {e}")

    # Estimate total windows
    if len(sample_files) > 0:
        avg_windows_per_file = total_windows_sampled / len(sample_files)
        estimated_total_windows = int(avg_windows_per_file * len(npy_files))
    else:
        estimated_total_windows = 0

    # Estimate memory requirements
    if detected_dtype and detected_features and estimated_total_windows > 0:
        bytes_per_element = np.dtype(detected_dtype).itemsize
        if detected_features:
            total_elements = estimated_total_windows * window_size * detected_features
        else:
            total_elements = estimated_total_windows * window_size
        estimated_memory_gb = (total_elements * bytes_per_element) / (1024 ** 3)
    else:
        estimated_memory_gb = 0

    logger.info(f"Estimation based on {len(sample_files)} files:")
    logger.info(f"  Estimated total windows: {estimated_total_windows:,}")
    logger.info(f"  Detected features: {detected_features}")
    logger.info(f"  Detected dtype: {detected_dtype}")
    logger.info(f"  Estimated memory requirement: {estimated_memory_gb:.2f} GB")

    return estimated_total_windows, estimated_memory_gb, detected_features, detected_dtype


def process_all_files_to_single_array(directory_path: Path, window_size: int = 512,
                                      step_size: int = 512, expected_features: Optional[int] = None,
                                      shuffle: bool = True, random_seed: int = None,
                                      max_memory_gb: float = 16.0) -> np.ndarray:
    """
    Process all numpy files and return a single concatenated array.

    Args:
        directory_path: Root directory containing .npy files
        window_size: Size of each window along the first dimension
        step_size: Step between windows along the first dimension
        expected_features: Expected number of features (numParents)
        shuffle: Whether to shuffle the final array
        random_seed: Random seed for shuffling
        max_memory_gb: Maximum memory to use (safety check)

    Returns:
        Single numpy array containing all windows
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    npy_files = find_numpy_files(directory_path)
    logger.info(f"Found {len(npy_files)} .npy files in {directory_path}")

    if not npy_files:
        logger.error("No .npy files found!")
        return np.array([])

    # Estimate requirements
    estimated_windows, estimated_memory_gb, detected_features, detected_dtype = estimate_total_windows(
        npy_files, window_size, step_size
    )

    if estimated_memory_gb > max_memory_gb:
        logger.error(f"Estimated memory requirement ({estimated_memory_gb:.2f} GB) "
                     f"exceeds maximum allowed ({max_memory_gb} GB)")
        logger.error("Consider using the batch version or increasing --max-memory-gb")
        raise MemoryError(f"Estimated memory requirement too high: {estimated_memory_gb:.2f} GB")

    # Validate expected features
    if expected_features is not None and detected_features is not None:
        if expected_features != detected_features:
            logger.warning(f"Expected {expected_features} features, detected {detected_features}")

    # Process all files and collect windows
    all_windows = []
    total_windows = 0
    processed_files = 0

    logger.info("Processing files and collecting windows...")

    for npy_file in tqdm(npy_files, desc="Processing files", unit="file"):
        try:
            # Load the array
            data = np.load(npy_file)
            original_dtype = data.dtype

            logger.debug(f"Loaded {npy_file.name}: {get_array_info(data)}")

            # Validate array dimensions and feature count
            if data.ndim == 2:
                current_features = data.shape[1]
                if detected_features is not None and current_features != detected_features:
                    logger.error(f"Feature count mismatch in {npy_file.name}: "
                                 f"expected {detected_features}, got {current_features}")
                    continue

            # Extract windows
            windows = extract_windows_2d(data, window_size, step_size)

            if len(windows) > 0:
                # Preserve original dtype
                windows = windows.astype(original_dtype)
                all_windows.append(windows)

                total_windows += len(windows)
                processed_files += 1

                logger.debug(f"Processed {npy_file.name}: {get_array_info(data)} -> "
                             f"{len(windows)} windows of shape {windows.shape[1:]}")
            else:
                if data.ndim >= 1:
                    required_size = window_size
                    actual_size = data.shape[0]
                    logger.warning(f"No windows extracted from {npy_file.name}: "
                                   f"file has {actual_size} time steps, need {required_size}")

        except Exception as e:
            logger.error(f"Error processing {npy_file}: {e}")

    logger.info(f"Processed {processed_files} files, extracted {total_windows} total windows")

    if not all_windows:
        logger.error("No windows were extracted!")
        return np.array([])

    # Concatenate all windows
    logger.info("Concatenating all windows...")
    try:
        final_array = np.concatenate(all_windows, axis=0)
        logger.info(f"Concatenated array shape: {final_array.shape}")
        logger.info(f"Final array: {get_array_info(final_array)}")
    except Exception as e:
        logger.error(f"Error concatenating arrays: {e}")
        raise

    # Shuffle if requested
    if shuffle and len(final_array) > 0:
        logger.info("Shuffling the final array...")
        indices = np.arange(len(final_array))
        np.random.shuffle(indices)
        final_array = final_array[indices]
        logger.info("Shuffling complete")

    return final_array


def save_single_file(array: np.ndarray, output_path: Path,
                     metadata_dict: dict = None) -> None:
    """
    Save array to a single file with metadata.

    Args:
        array: Array to save
        output_path: Output file path
        metadata_dict: Optional metadata to save alongside
    """
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the main array
    logger.info(f"Saving array to {output_path}...")
    np.save(output_path, array)

    # Save metadata
    if metadata_dict:
        metadata_path = output_path.with_suffix('.txt')
        logger.info(f"Saving metadata to {metadata_path}...")

        with open(metadata_path, 'w') as f:
            for key, value in metadata_dict.items():
                f.write(f"{key}: {value}\n")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {output_path.name}: {get_array_info(array)}, file size: {file_size_mb:.2f}MB")


def main():
    parser = argparse.ArgumentParser(
        description="Extract windowed data from 2D numpy arrays and save as single file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - save all windows to single file
  python extract_windows_single.py /path/to/input /path/to/output/windows.npy

  # With overlapping windows
  python extract_windows_single.py /path/to/input /path/to/output/windows.npy --step-size 256

  # Without shuffling (preserve file order)
  python extract_windows_single.py /path/to/input /path/to/output/windows.npy --no-shuffle

  # Increase memory limit for large datasets
  python extract_windows_single.py /path/to/input /path/to/output/windows.npy --max-memory-gb 32
        """
    )
    parser.add_argument("--input_dir", type=str,
                        help="Input directory containing .npy files")
    parser.add_argument("--output_file", type=str,
                        help="Output .npy file path")
    parser.add_argument("--window-size", type=int, default=512,
                        help="Size of each window along first dimension (default: 512)")
    parser.add_argument("--step-size", type=int, default=512,
                        help="Step between windows along first dimension (default: 512)")
    parser.add_argument("--expected-features", type=int, default=None,
                        help="Expected number of features (numParents). Auto-detected if not specified.")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Don't shuffle the final array (preserve file order)")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    parser.add_argument("--max-memory-gb", type=float, default=16.0,
                        help="Maximum memory to use in GB (default: 16.0)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input_dir)
    output_path = Path(args.output_file)

    # Validate input directory
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return 1

    # Ensure output file has .npy extension
    if not output_path.suffix == '.npy':
        output_path = output_path.with_suffix('.npy')
        logger.info(f"Added .npy extension: {output_path}")

    logger.info(f"Starting processing...")
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Window size: {args.window_size}")
    logger.info(f"Step size: {args.step_size}")
    logger.info(f"Expected features: {args.expected_features or 'auto-detect'}")
    logger.info(f"Shuffle: {not args.no_shuffle}")
    logger.info(f"Random seed: {args.random_seed}")
    logger.info(f"Max memory: {args.max_memory_gb} GB")

    try:
        # Process all files
        final_array = process_all_files_to_single_array(
            input_path,
            args.window_size,
            args.step_size,
            args.expected_features,
            shuffle=not args.no_shuffle,
            random_seed=args.random_seed,
            max_memory_gb=args.max_memory_gb
        )

        if len(final_array) == 0:
            logger.error("No data was processed!")
            return 1

        # Prepare metadata
        metadata = {
            'input_directory': str(input_path),
            'total_windows': len(final_array),
            'window_size': args.window_size,
            'step_size': args.step_size,
            'window_shape': str(final_array.shape[1:]) if len(final_array) > 0 else 'unknown',
            'final_array_shape': str(final_array.shape),
            'shuffled': not args.no_shuffle,
            'random_seed': args.random_seed if not args.no_shuffle else 'N/A',
            'dtype': str(final_array.dtype)
        }

        # Save the result
        save_single_file(final_array, output_path, metadata)

        logger.info("Processing completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())