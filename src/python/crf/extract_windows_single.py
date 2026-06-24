#!/usr/bin/env python3
"""
Extract windows from numpy arrays in nested directories and save as single file.
Handles arrays of shape (time_steps, numParents) or (time_steps, numParents+1) and ensures
output windows have shape (window_size, numParents+2).

This version includes comprehensive debugging and validation to ensure data integrity.
"""

import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import random
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def debug_array_format(data: np.ndarray, filename: str = "", max_rows: int = 5):
    """Debug function to inspect array format."""
    logger.info(f"Debugging array {filename}:")
    logger.info(f"  Shape: {data.shape}")
    logger.info(f"  Dtype: {data.dtype}")

    if data.ndim == 2:
        # Show statistics for each column
        for col in range(min(data.shape[1], 10)):  # Show first 10 columns
            col_data = data[:, col]
            unique_vals = np.unique(col_data)
            logger.info(f"  Column {col}: min={col_data.min():.3f}, max={col_data.max():.3f}, "
                        f"unique_count={len(unique_vals)}")

            # If column is mostly zeros, that's suspicious
            zero_count = np.sum(col_data == 0)
            zero_pct = zero_count / len(col_data) * 100
            if zero_pct > 90:
                logger.warning(f"    Column {col} is {zero_pct:.1f}% zeros!")

        # Show a few sample rows
        logger.info(f"  Sample rows:")
        for i in range(min(max_rows, data.shape[0])):
            logger.info(f"    Row {i}: {data[i]}")


def check_zero_positions_in_windows(windows: np.ndarray, window_descriptions: str = ""):
    """
    Check for positions within windows where all features are zero.

    Args:
        windows: Array of shape (n_windows, window_size, n_features)
        window_descriptions: Description for logging

    Returns:
        List of (window_idx, position_idx) tuples where all features are zero
    """
    logger.info(f"Checking for zero positions in {window_descriptions}...")

    zero_positions = []

    for win_idx in range(windows.shape[0]):
        for pos_idx in range(windows.shape[1]):  # For each position in the window
            position_data = windows[win_idx, pos_idx, :]  # All features at this position

            if np.all(position_data == 0):
                zero_positions.append((win_idx, pos_idx))
                if len(zero_positions) <= 10:  # Only log first 10 to avoid spam
                    logger.error(f"Position {pos_idx} in window {win_idx} has ALL ZERO features: {position_data}")

    if zero_positions:
        logger.error(f"Found {len(zero_positions)} positions with all-zero features!")
    else:
        logger.info(f"✅ No positions found with all-zero features in {len(windows)} windows")

    return zero_positions


def standardize_array_format(data: np.ndarray, num_parents: int) -> np.ndarray:
    """
    Standardize array format to have numParents+2 columns.

    Args:
        data: Input array of shape (time_steps, features)
        num_parents: Number of parent columns expected

    Returns:
        Array with shape (time_steps, num_parents+2)
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D array with shape {data.shape}")

    time_steps, n_features = data.shape

    logger.debug(f"Standardizing array: input shape {data.shape}, expecting {num_parents} parents")

    if n_features == num_parents + 1:
        # Haploid case: duplicate the last column to make diploid
        parents_data = data[:, :num_parents]  # First numParents columns
        haploid_label = data[:, -1:]  # Last column

        # Debug the label column
        unique_labels = np.unique(haploid_label)
        logger.debug(f"Haploid labels - unique values: {unique_labels}")

        # Duplicate haploid label to create diploid labels
        diploid_labels = np.concatenate([haploid_label, haploid_label], axis=1)

        # Combine: parents + diploid_labels
        result = np.concatenate([parents_data, diploid_labels], axis=1)
        logger.debug(f"Converted haploid format {data.shape} -> diploid format {result.shape}")

    elif n_features == num_parents + 2:
        # Already diploid case: use as-is
        result = data

        # Debug the label columns
        label1_unique = np.unique(data[:, -2])
        label2_unique = np.unique(data[:, -1])
        logger.debug(f"Diploid labels - label1 unique: {label1_unique}, label2 unique: {label2_unique}")
        logger.debug(f"Already diploid format: {data.shape}")

    elif n_features == num_parents:
        # No labels case: this shouldn't happen for labeled data
        raise ValueError(
            f"Array has no label columns: expected {num_parents}+1 or {num_parents}+2 features, got {n_features}")

    else:
        raise ValueError(
            f"Unexpected number of features: expected {num_parents}+1 or {num_parents}+2, got {n_features}")

    # Verify no positions have all zeros (this would be problematic)
    zero_positions = []
    for i in range(min(1000, result.shape[0])):  # Check first 1000 positions
        if np.all(result[i, :] == 0):
            zero_positions.append(i)

    if zero_positions:
        logger.error(f"Found {len(zero_positions)} positions with all-zero features in standardized data!")
        logger.error(f"Zero positions: {zero_positions[:10]}...")  # Show first 10

    return result


def extract_windows_2d(data: np.ndarray, window_size: int = 512, step_size: int = 512,
                       num_parents: int = 24, debug: bool = False) -> np.ndarray:
    """
    Extract windows from 2D array and standardize format.

    Args:
        data: Input numpy array of shape (time_steps, features)
        window_size: Size of each window along the first dimension
        step_size: Step between windows along the first dimension
        num_parents: Number of parent columns
        debug: Whether to add debug output

    Returns:
        Array of windows with shape (n_windows, window_size, num_parents+2)
    """
    if debug:
        logger.info(f"Input data shape: {data.shape}")

        # Check for zero positions in input data
        logger.info("Checking input data for positions with all-zero features...")
        zero_input_positions = []
        for i in range(min(100, data.shape[0])):  # Check first 100 positions
            if np.all(data[i, :] == 0):
                zero_input_positions.append(i)

        if zero_input_positions:
            logger.error(f"Input data has {len(zero_input_positions)} positions with all-zero features!")
        else:
            logger.info("✅ No positions with all-zero features in input data")

    # First standardize the array format
    standardized_data = standardize_array_format(data, num_parents)

    if debug:
        logger.info(f"Standardized data shape: {standardized_data.shape}")

    time_steps, num_features = standardized_data.shape

    if time_steps < window_size:
        logger.warning(f"Data too short: {time_steps} < {window_size}")
        return np.array([]).reshape(0, window_size, num_features)

    # Calculate number of windows
    num_windows = (time_steps - window_size) // step_size + 1

    if debug:
        logger.info(f"Will extract {num_windows} windows")

    # Extract windows using simple slicing (no sliding_window_view complications)
    windows = []
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window = standardized_data[start_idx:end_idx].copy()  # Shape: (window_size, num_features)

        if debug and i < 3:  # Debug first few windows
            logger.info(f"Window {i} indices: [{start_idx}:{end_idx}]")
            logger.info(f"Window {i} shape: {window.shape}")

            # Check each position in this window for all-zero features
            zero_positions_in_window = []
            for pos in range(window.shape[0]):
                if np.all(window[pos, :] == 0):
                    zero_positions_in_window.append(pos)

            if zero_positions_in_window:
                logger.error(f"Window {i} has {len(zero_positions_in_window)} positions with all-zero features!")
                for pos in zero_positions_in_window[:3]:  # Show first 3
                    global_pos = start_idx + pos
                    logger.error(f"  Position {pos} (global {global_pos}): {window[pos, :]}")
            else:
                logger.info(f"✅ Window {i} has no positions with all-zero features")

        windows.append(window)

    if windows:
        result = np.stack(windows, axis=0)  # Shape: (n_windows, window_size, num_features)

        if debug or True:  # Always check this in production
            logger.info(f"Final result shape: {result.shape}")

            # Check final result for positions with all-zero features
            zero_positions = check_zero_positions_in_windows(result, "EXTRACTED WINDOWS")

            if zero_positions:
                logger.error(f"CRITICAL: Found {len(zero_positions)} positions with all-zero features in final result!")
                return np.array([]).reshape(0, window_size, num_features)  # Return empty on error

        return result
    else:
        return np.array([]).reshape(0, window_size, num_features)


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
                           num_parents: int, sample_size: int = 10) -> tuple:
    """
    Estimate the total number of windows and memory requirements by sampling files.

    Args:
        npy_files: List of numpy file paths
        window_size: Size of each window
        step_size: Step between windows
        num_parents: Number of parent columns
        sample_size: Number of files to sample for estimation

    Returns:
        Tuple of (estimated_total_windows, estimated_memory_gb, detected_dtype)
    """
    logger.info("Estimating total windows and memory requirements...")

    sample_files = npy_files[:min(sample_size, len(npy_files))]
    total_windows_sampled = 0
    detected_dtype = None

    for npy_file in sample_files:
        try:
            data = np.load(npy_file)
            if detected_dtype is None:
                detected_dtype = data.dtype

            windows = extract_windows_2d(data, window_size, step_size, num_parents)
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
    if detected_dtype and estimated_total_windows > 0:
        bytes_per_element = np.dtype(detected_dtype).itemsize
        total_elements = estimated_total_windows * window_size * (num_parents + 2)
        estimated_memory_gb = (total_elements * bytes_per_element) / (1024 ** 3)
    else:
        estimated_memory_gb = 0

    logger.info(f"Estimation based on {len(sample_files)} files:")
    logger.info(f"  Estimated total windows: {estimated_total_windows:,}")
    logger.info(f"  Output shape per window: ({window_size}, {num_parents + 2})")
    logger.info(f"  Detected dtype: {detected_dtype}")
    logger.info(f"  Estimated memory requirement: {estimated_memory_gb:.2f} GB")

    return estimated_total_windows, estimated_memory_gb, detected_dtype


def test_single_file(file_path: Path, num_parents: int, window_size: int = 512, step_size: int = 512):
    """Test window extraction on a single file with detailed debugging."""
    logger.info(f"Testing file: {file_path}")

    # Load original data
    data = np.load(file_path)
    logger.info(f"Loaded data shape: {data.shape}")
    debug_array_format(data, str(file_path.name))

    # Extract windows with debugging
    windows = extract_windows_2d(data, window_size, step_size, num_parents, debug=True)

    if len(windows) > 0:
        logger.info(f"Successfully extracted {len(windows)} windows")

        # Test a few windows
        for i in range(min(3, len(windows))):
            window = windows[i]
            logger.info(f"Window {i} shape: {window.shape}")

            # Check labels
            labels = window[:, num_parents:]
            unique_labels_col1 = np.unique(labels[:, 0])
            unique_labels_col2 = np.unique(labels[:, 1]) if labels.shape[1] > 1 else []

            logger.info(f"Window {i} label column 1 unique values: {unique_labels_col1}")
            logger.info(f"Window {i} label column 2 unique values: {unique_labels_col2}")

    return windows


def comprehensive_debug_test(file_path: Path, num_parents: int, window_size: int = 512, step_size: int = 512):
    """Comprehensive debugging to find positions with all-zero features."""
    logger.info(f"=== COMPREHENSIVE DEBUG TEST FOR ZERO POSITIONS ===")
    logger.info(f"File: {file_path}")

    # Load and inspect original data
    data = np.load(file_path)
    logger.info(f"Original data shape: {data.shape}")

    # Check for positions with all-zero features in original data
    logger.info("Checking original data for positions with all-zero features...")
    zero_positions_original = []

    for i in range(min(10000, data.shape[0])):  # Check first 10k positions
        if np.all(data[i, :] == 0):
            zero_positions_original.append(i)

    if zero_positions_original:
        logger.error(f"Found {len(zero_positions_original)} positions with all-zero features in ORIGINAL data!")
        logger.error(f"First few zero positions: {zero_positions_original[:20]}")
    else:
        logger.info("✅ No positions with all-zero features found in original data")

    # Now run the actual extraction
    logger.info("Running actual window extraction...")
    windows = extract_windows_2d(data, window_size, step_size, num_parents, debug=True)

    return windows


def process_all_files_to_single_array(directory_path: Path, window_size: int = 512,
                                      step_size: int = 512, num_parents: int = 24,
                                      shuffle: bool = True, random_seed: int = None,
                                      max_memory_gb: float = 16.0) -> np.ndarray:
    """
    Process all numpy files and return a single concatenated array.

    Args:
        directory_path: Root directory containing .npy files
        window_size: Size of each window along the first dimension
        step_size: Step between windows along the first dimension
        num_parents: Number of parent columns
        shuffle: Whether to shuffle the final array
        random_seed: Random seed for shuffling
        max_memory_gb: Maximum memory to use (safety check)

    Returns:
        Single numpy array containing all windows with shape (total_windows, window_size, num_parents+2)
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
    estimated_windows, estimated_memory_gb, detected_dtype = estimate_total_windows(
        npy_files, window_size, step_size, num_parents
    )

    if estimated_memory_gb > max_memory_gb:
        logger.error(f"Estimated memory requirement ({estimated_memory_gb:.2f} GB) "
                     f"exceeds maximum allowed ({max_memory_gb} GB)")
        logger.error("Consider using the batch version or increasing --max-memory-gb")
        raise MemoryError(f"Estimated memory requirement too high: {estimated_memory_gb:.2f} GB")

    # Process all files and collect windows
    all_windows = []
    total_windows = 0
    processed_files = 0
    failed_files = 0

    logger.info("Processing files and collecting windows...")

    for npy_file in tqdm(npy_files, desc="Processing files", unit="file"):
        try:
            # Load the array
            data = np.load(npy_file)
            original_dtype = data.dtype

            logger.debug(f"Loaded {npy_file.name}: {get_array_info(data)}")

            # Extract windows (this also standardizes the format)
            windows = extract_windows_2d(data, window_size, step_size, num_parents)

            if len(windows) > 0:
                # Preserve original dtype
                windows = windows.astype(original_dtype)
                all_windows.append(windows)

                total_windows += len(windows)
                processed_files += 1

                logger.debug(f"Processed {npy_file.name}: {get_array_info(data)} -> "
                             f"{len(windows)} windows of shape {windows.shape[1:]}")
            else:
                required_size = window_size
                actual_size = data.shape[0] if len(data.shape) > 0 else 0
                logger.warning(f"No windows extracted from {npy_file.name}: "
                               f"file has {actual_size} time steps, need {required_size}")

        except Exception as e:
            logger.error(f"Error processing {npy_file}: {e}")
            failed_files += 1

    logger.info(f"Processed {processed_files} files successfully, {failed_files} files failed")
    logger.info(f"Extracted {total_windows} total windows")

    if not all_windows:
        logger.error("No windows were extracted!")
        return np.array([])

    # Concatenate all windows
    logger.info("Concatenating all windows...")
    try:
        final_array = np.concatenate(all_windows, axis=0)
        logger.info(f"Concatenated array shape: {final_array.shape}")
        logger.info(f"Final array: {get_array_info(final_array)}")

        # Verify final shape
        expected_shape = (total_windows, window_size, num_parents + 2)
        if final_array.shape != expected_shape:
            logger.warning(f"Final shape {final_array.shape} differs from expected {expected_shape}")

        # CRITICAL: Check for zero positions in final result
        logger.info("Performing final validation...")
        zero_positions = check_zero_positions_in_windows(final_array, "FINAL CONCATENATED RESULT")

        if zero_positions:
            logger.error(
                f"CRITICAL ERROR: Final result contains {len(zero_positions)} positions with all-zero features!")
            logger.error("This indicates a serious problem with the data or processing!")
            # You might want to raise an exception here or return empty array
            # raise ValueError("Final result contains invalid zero positions!")

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
  # Basic usage with 24 parents
  python extract_windows_single.py /path/to/input /path/to/output/windows.npy --num-parents 24

  # With overlapping windows
  python extract_windows_single.py /path/to/input /path/to/output/windows.npy --num-parents 24 --step-size 256

  # Test single file for debugging
  python extract_windows_single.py --test-file /path/to/file.npy --num-parents 24

  # Comprehensive debugging
  python extract_windows_single.py --comprehensive-debug /path/to/file.npy --num-parents 24
        """
    )
    parser.add_argument("--input_dir", type=str, nargs='?',
                        help="Input directory containing .npy files")
    parser.add_argument("--output_file", type=str, nargs='?',
                        help="Output .npy file path")
    parser.add_argument("--num-parents", type=int, required=True,
                        help="Number of parent columns in the data")
    parser.add_argument("--window-size", type=int, default=512,
                        help="Size of each window along first dimension (default: 512)")
    parser.add_argument("--step-size", type=int, default=512,
                        help="Step between windows along first dimension (default: 512)")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Don't shuffle the final array (preserve file order)")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    parser.add_argument("--max-memory-gb", type=float, default=16.0,
                        help="Maximum memory to use in GB (default: 16.0)")
    parser.add_argument("--test-file", type=str, default=None,
                        help="Test window extraction on a single file (debugging)")
    parser.add_argument("--comprehensive-debug", type=str, default=None,
                        help="Run comprehensive debugging on a single file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle test modes
    if args.test_file:
        test_file_path = Path(args.test_file)
        if not test_file_path.exists():
            logger.error(f"Test file does not exist: {test_file_path}")
            return 1

        logger.info("=== SINGLE FILE TEST MODE ===")
        windows = test_single_file(test_file_path, args.num_parents, args.window_size, args.step_size)

        if len(windows) > 0:
            logger.info("✅ Single file test completed successfully")
        else:
            logger.error("✗ Single file test failed")
        return 0

    if args.comprehensive_debug:
        debug_file_path = Path(args.comprehensive_debug)
        if not debug_file_path.exists():
            logger.error(f"Debug file does not exist: {debug_file_path}")
            return 1

        logger.info("=== COMPREHENSIVE DEBUG MODE ===")
        windows = comprehensive_debug_test(debug_file_path, args.num_parents, args.window_size, args.step_size)

        if len(windows) > 0:
            zero_positions = check_zero_positions_in_windows(windows, "DEBUG RESULT")
            logger.info(f"Debug complete: {len(zero_positions)} zero positions found")

        return 0

    # Normal processing mode
    if not args.input_dir or not args.output_file:
        logger.error("Input directory and output file are required for normal processing")
        parser.print_help()
        return 1

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
    logger.info(f"Number of parents: {args.num_parents}")
    logger.info(f"Expected output shape per window: ({args.window_size}, {args.num_parents + 2})")
    logger.info(f"Window size: {args.window_size}")
    logger.info(f"Step size: {args.step_size}")
    logger.info(f"Shuffle: {not args.no_shuffle}")
    logger.info(f"Random seed: {args.random_seed}")
    logger.info(f"Max memory: {args.max_memory_gb} GB")

    try:
        # Process all files
        final_array = process_all_files_to_single_array(
            input_path,
            args.window_size,
            args.step_size,
            args.num_parents,
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
            'num_parents': args.num_parents,
            'total_windows': len(final_array),
            'window_size': args.window_size,
            'step_size': args.step_size,
            'final_array_shape': str(final_array.shape),
            'window_shape': f"({args.window_size}, {args.num_parents + 2})",
            'shuffled': not args.no_shuffle,
            'random_seed': args.random_seed if not args.no_shuffle else 'N/A',
            'dtype': str(final_array.dtype),
            'label_format': 'diploid (duplicated from haploid if needed)'
        }

        # Save the result
        save_single_file(final_array, output_path, metadata)

        logger.info("Processing completed successfully!")
        logger.info(f"Output shape: {final_array.shape}")
        logger.info(f"Each window shape: {final_array.shape[1:]}")

        return 0

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())