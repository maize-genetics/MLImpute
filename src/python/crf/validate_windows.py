#!/usr/bin/env python3
"""
Validation script for extracted windows to check:
1. No positions with all-zero features
2. Label consistency (accounting for diploid nature)
3. General data integrity
"""

import numpy as np
import argparse
from pathlib import Path
import logging
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# def analyze_sharing_distribution(data: np.ndarray, num_parents: int, plot_output: Path = None) -> dict:
#     """
#     Analyze the distribution of non-zero entries (sharing) in parent columns across ALL positions.
#
#     Args:
#         data: Array of shape (n_windows, window_size, n_features)
#         num_parents: Number of parent columns
#         plot_output: Optional path to save distribution plot
#
#     Returns:
#         Dictionary with sharing analysis results
#     """
#     logger.info("=== ANALYZING SHARING DISTRIBUTION ===")
#
#     # Extract parent data only (exclude labels)
#     parent_data = data[:, :, :num_parents]  # Shape: (n_windows, window_size, num_parents)
#
#     sharing_counts = []
#
#     # For each window, for each position, count non-zero entries
#     for win_idx in range(parent_data.shape[0]):
#         for pos_idx in range(parent_data.shape[1]):
#             position_parents = parent_data[win_idx, pos_idx, :]  # Shape: (num_parents,)
#
#             # Count non-zero entries
#             non_zero_count = np.sum(position_parents != 0)
#             sharing_counts.append(non_zero_count)
#
#     # Convert to numpy array for easier analysis
#     sharing_counts = np.array(sharing_counts)
#
#     # Calculate statistics
#     total_positions = len(sharing_counts)
#     unique_counts, count_frequencies = np.unique(sharing_counts, return_counts=True)
#
#     stats = {
#         'total_positions': total_positions,
#         'min_sharing': int(sharing_counts.min()),
#         'max_sharing': int(sharing_counts.max()),
#         'mean_sharing': float(sharing_counts.mean()),
#         'median_sharing': float(np.median(sharing_counts)),
#         'std_sharing': float(sharing_counts.std()),
#         'sharing_distribution': dict(zip([int(x) for x in unique_counts],
#                                          [int(x) for x in count_frequencies])),
#         'sharing_percentages': dict(zip([int(x) for x in unique_counts],
#                                         [float(x / total_positions * 100) for x in count_frequencies]))
#     }
#
#     # Log basic statistics
#     logger.info(f"Sharing distribution analysis on {total_positions:,} positions:")
#     logger.info(f"  Sharing range: [{stats['min_sharing']}, {stats['max_sharing']}]")
#     logger.info(f"  Mean sharing: {stats['mean_sharing']:.2f}")
#     logger.info(f"  Median sharing: {stats['median_sharing']:.1f}")
#     logger.info(f"  Std deviation: {stats['std_sharing']:.2f}")
#
#     # Show distribution
#     logger.info("  Sharing distribution:")
#     for sharing_count in sorted(unique_counts):
#         count = stats['sharing_distribution'][int(sharing_count)]
#         percentage = stats['sharing_percentages'][int(sharing_count)]
#         logger.info(f"    {int(sharing_count)} parents: {count:,} positions ({percentage:.1f}%)")
#
#     # Check for potential issues
#     warnings = []
#
#     # Check if too many positions have 0 sharing (all zeros in parents)
#     zero_sharing_pct = stats['sharing_percentages'].get(0, 0)
#     if zero_sharing_pct > 5:  # More than 5% positions have no sharing
#         warnings.append(f"High percentage of positions with no sharing: {zero_sharing_pct:.1f}%")
#
#     # Check if sharing is too concentrated
#     max_concentration = max(stats['sharing_percentages'].values())
#     if max_concentration > 80:  # More than 80% positions have same sharing count
#         most_common_sharing = max(stats['sharing_percentages'].keys(),
#                                   key=lambda k: stats['sharing_percentages'][k])
#         warnings.append(f"Sharing too concentrated: {max_concentration:.1f}% at {most_common_sharing} parents")
#
#     # Check if sharing distribution looks reasonable
#     if stats['mean_sharing'] < 1:
#         warnings.append(f"Very low average sharing: {stats['mean_sharing']:.2f}")
#     elif stats['mean_sharing'] > num_parents * 0.8:
#         warnings.append(f"Very high average sharing: {stats['mean_sharing']:.2f} (>{num_parents * 0.8:.1f})")
#
#     stats['warnings'] = warnings
#     stats['passed'] = len(warnings) == 0
#
#     if warnings:
#         logger.warning("⚠️  Potential issues with sharing distribution:")
#         for warning in warnings:
#             logger.warning(f"  - {warning}")
#     else:
#         logger.info("✅ Sharing distribution looks normal")
#
#     # Create visualization if requested
#     if plot_output:
#         create_sharing_plots(sharing_counts, stats, plot_output, num_parents, parent_data)
#
#     return stats
#
#
# def create_sharing_plots(sharing_counts: np.ndarray, stats: dict, output_path: Path, num_parents: int,
#                          parent_data: np.ndarray = None):
#     """
#     Create simplified visualization plots for sharing distribution.
#
#     Args:
#         sharing_counts: Array of sharing counts for all positions
#         stats: Statistics dictionary from analyze_sharing_distribution
#         output_path: Path to save the plot
#         num_parents: Number of parent columns
#         parent_data: Optional parent data array to calculate count-scaled version
#     """
#     logger.info(f"Creating sharing distribution plots...")
#
#     # Set up the plotting style
#     plt.style.use('default')
#
#     # Determine if we need 1 or 2 plots
#     if parent_data is not None:
#         fig, axes = plt.subplots(1, 2, figsize=(15, 6))
#         ax1, ax2 = axes
#     else:
#         fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
#         ax2 = None
#
#     fig.suptitle(f'Sharing Distribution Analysis\n({len(sharing_counts):,} positions, {num_parents} parents)',
#                  fontsize=16, fontweight='bold')
#
#     # Plot 1: Non-zero frequency histogram (original)
#     bins = np.arange(stats['min_sharing'], stats['max_sharing'] + 2) - 0.5
#     counts, _, patches = ax1.hist(sharing_counts, bins=bins, alpha=0.7, color='skyblue',
#                                   edgecolor='black', density=False)
#
#     ax1.set_xlabel('Number of Non-Zero Parents')
#     ax1.set_ylabel('Frequency (Number of Positions)')
#     ax1.set_title('Non-Zero Parent Count Distribution')
#     ax1.grid(True, alpha=0.3)
#
#     # Add percentage labels on bars
#     total_positions = len(sharing_counts)
#     for i, (count, patch) in enumerate(zip(counts, patches)):
#         if count > 0:
#             percentage = count / total_positions * 100
#             if percentage > 1:  # Only label bars with >1%
#                 height = patch.get_height()
#                 ax1.text(patch.get_x() + patch.get_width() / 2., height + height * 0.01,
#                          f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
#
#     # Add statistics text
#     stats_text = f'Mean: {stats["mean_sharing"]:.2f}\nMedian: {stats["median_sharing"]:.1f}\nStd: {stats["std_sharing"]:.2f}'
#     ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
#              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#
#     # Plot 2: Count-scaled histogram (if parent data provided)
#     if parent_data is not None and ax2 is not None:
#         # Calculate count-scaled sharing
#         scaled_sharing_counts = []
#
#         for win_idx in range(parent_data.shape[0]):
#             for pos_idx in range(parent_data.shape[1]):
#                 position_parents = parent_data[win_idx, pos_idx, :]
#
#                 # Count non-zero entries
#                 non_zero_count = np.sum(position_parents != 0)
#
#                 # Get the actual sum of counts (scale factor)
#                 if non_zero_count > 0:
#                     # Use the sum of non-zero values as the scale factor
#                     scale_factor = np.sum(position_parents[position_parents != 0])
#                     scaled_sharing_counts.extend([non_zero_count] * int(scale_factor))
#                 # If all zeros, don't contribute to the scaled distribution
#
#         scaled_sharing_counts = np.array(scaled_sharing_counts)
#
#         if len(scaled_sharing_counts) > 0:
#             # Calculate bins for scaled data
#             scaled_min = scaled_sharing_counts.min()
#             scaled_max = scaled_sharing_counts.max()
#             scaled_bins = np.arange(scaled_min, scaled_max + 2) - 0.5
#
#             scaled_counts, _, scaled_patches = ax2.hist(scaled_sharing_counts, bins=scaled_bins,
#                                                         alpha=0.7, color='lightcoral',
#                                                         edgecolor='black', density=False)
#
#             ax2.set_xlabel('Number of Non-Zero Parents')
#             ax2.set_ylabel('Frequency (Scaled by Count Values)')
#             ax2.set_title('Count-Scaled Distribution')
#             ax2.grid(True, alpha=0.3)
#
#             # Add percentage labels on bars for scaled plot
#             total_scaled = len(scaled_sharing_counts)
#             for i, (count, patch) in enumerate(zip(scaled_counts, scaled_patches)):
#                 if count > 0:
#                     percentage = count / total_scaled * 100
#                     if percentage > 1:  # Only label bars with >1%
#                         height = patch.get_height()
#                         ax2.text(patch.get_x() + patch.get_width() / 2., height + height * 0.01,
#                                  f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
#
#             # Add statistics for scaled data
#             scaled_stats_text = f'Mean: {scaled_sharing_counts.mean():.2f}\nMedian: {np.median(scaled_sharing_counts):.1f}\nStd: {scaled_sharing_counts.std():.2f}'
#             ax2.text(0.02, 0.98, scaled_stats_text, transform=ax2.transAxes, verticalalignment='top',
#                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#         else:
#             ax2.text(0.5, 0.5, 'No data for count-scaled plot\n(all positions have zero counts)',
#                      ha='center', va='center', transform=ax2.transAxes,
#                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
#             ax2.set_xlabel('Number of Non-Zero Parents')
#             ax2.set_ylabel('Frequency (Scaled by Count Values)')
#             ax2.set_title('Count-Scaled Distribution')
#
#     # Adjust layout and save
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()
#
#     logger.info(f"Sharing distribution plots saved to {output_path}")


def analyze_sharing_distribution(data: np.ndarray, num_parents: int, plot_output: Path = None) -> dict:
    """
    Analyze the distribution of non-zero entries (sharing) in parent columns across ALL positions.
    Single-pass efficient version using dictionaries.

    Args:
        data: Array of shape (n_windows, window_size, n_features)
        num_parents: Number of parent columns
        plot_output: Optional path to save distribution plot

    Returns:
        Dictionary with sharing analysis results
    """
    logger.info("=== ANALYZING SHARING DISTRIBUTION ===")

    # Extract parent data only (exclude labels)
    parent_data = data[:, :, :num_parents]  # Shape: (n_windows, window_size, num_parents)

    # Use dictionaries for efficient counting
    sharing_counts_dict = {}  # {sharing_count: frequency}
    scaled_counts_dict = {}  # {sharing_count: scaled_frequency}

    total_positions = 0
    sharing_sum = 0  # For mean calculation
    sharing_values = []  # For median/std calculation (only store values, not repeated)

    # Single pass through all positions
    for win_idx in range(parent_data.shape[0]):
        for pos_idx in range(parent_data.shape[1]):
            position_parents = parent_data[win_idx, pos_idx, :]  # Shape: (num_parents,)

            # Count non-zero entries
            non_zero_mask = position_parents != 0
            non_zero_count = np.sum(non_zero_mask)

            # Update regular sharing counts
            sharing_counts_dict[non_zero_count] = sharing_counts_dict.get(non_zero_count, 0) + 1

            # Calculate scale factor (sum of non-zero values)
            if non_zero_count > 0:
                scale_factor = int(np.sum(position_parents[non_zero_mask]))
                scaled_counts_dict[non_zero_count] = scaled_counts_dict.get(non_zero_count, 0) + scale_factor

            # For statistics calculation
            total_positions += 1
            sharing_sum += non_zero_count
            sharing_values.append(non_zero_count)

    # Convert to arrays for statistics that need them
    sharing_values = np.array(sharing_values)

    # Calculate statistics
    mean_sharing = sharing_sum / total_positions if total_positions > 0 else 0
    median_sharing = np.median(sharing_values) if len(sharing_values) > 0 else 0
    std_sharing = np.std(sharing_values) if len(sharing_values) > 0 else 0

    min_sharing = min(sharing_counts_dict.keys()) if sharing_counts_dict else 0
    max_sharing = max(sharing_counts_dict.keys()) if sharing_counts_dict else 0

    # Convert to percentages
    sharing_percentages = {count: (freq / total_positions * 100)
                           for count, freq in sharing_counts_dict.items()}

    # Calculate scaled statistics
    total_scaled = sum(scaled_counts_dict.values()) if scaled_counts_dict else 0
    scaled_percentages = {count: (freq / total_scaled * 100)
                          for count, freq in scaled_counts_dict.items()} if total_scaled > 0 else {}

    stats = {
        'total_positions': total_positions,
        'min_sharing': min_sharing,
        'max_sharing': max_sharing,
        'mean_sharing': float(mean_sharing),
        'median_sharing': float(median_sharing),
        'std_sharing': float(std_sharing),
        'sharing_distribution': sharing_counts_dict,
        'sharing_percentages': sharing_percentages,
        'scaled_distribution': scaled_counts_dict,
        'scaled_percentages': scaled_percentages,
        'total_scaled': total_scaled
    }

    # Log basic statistics
    logger.info(f"Sharing distribution analysis on {total_positions:,} positions:")
    logger.info(f"  Sharing range: [{min_sharing}, {max_sharing}]")
    logger.info(f"  Mean sharing: {mean_sharing:.2f}")
    logger.info(f"  Median sharing: {median_sharing:.1f}")
    logger.info(f"  Std deviation: {std_sharing:.2f}")

    # Show distribution
    logger.info("  Regular sharing distribution:")
    for sharing_count in sorted(sharing_counts_dict.keys()):
        count = sharing_counts_dict[sharing_count]
        percentage = sharing_percentages[sharing_count]
        logger.info(f"    {sharing_count} parents: {count:,} positions ({percentage:.1f}%)")

    logger.info("  Count-scaled distribution:")
    for sharing_count in sorted(scaled_counts_dict.keys()):
        scaled_count = scaled_counts_dict[sharing_count]
        scaled_percentage = scaled_percentages[sharing_count]
        logger.info(f"    {sharing_count} parents: {scaled_count:,} scaled ({scaled_percentage:.1f}%)")

    # Check for potential issues
    warnings = []

    # Check if too many positions have 0 sharing
    zero_sharing_pct = sharing_percentages.get(0, 0)
    if zero_sharing_pct > 5:
        warnings.append(f"High percentage of positions with no sharing: {zero_sharing_pct:.1f}%")

    # Check if sharing is too concentrated
    if sharing_percentages:
        max_concentration = max(sharing_percentages.values())
        if max_concentration > 80:
            most_common_sharing = max(sharing_percentages.keys(),
                                      key=lambda k: sharing_percentages[k])
            warnings.append(f"Sharing too concentrated: {max_concentration:.1f}% at {most_common_sharing} parents")

    # Check if sharing distribution looks reasonable
    if mean_sharing < 1:
        warnings.append(f"Very low average sharing: {mean_sharing:.2f}")
    elif mean_sharing > num_parents * 0.8:
        warnings.append(f"Very high average sharing: {mean_sharing:.2f} (>{num_parents * 0.8:.1f})")

    stats['warnings'] = warnings
    stats['passed'] = len(warnings) == 0

    if warnings:
        logger.warning("⚠️  Potential issues with sharing distribution:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    else:
        logger.info("✅ Sharing distribution looks normal")

    # Create visualization if requested
    if plot_output:
        create_sharing_plots_efficient(stats, plot_output, num_parents)

    return stats


def create_sharing_plots_efficient(stats: dict, output_path: Path, num_parents: int):
    """
    Create simplified visualization plots using pre-computed statistics.

    Args:
        stats: Statistics dictionary from analyze_sharing_distribution
        output_path: Path to save the plot
        num_parents: Number of parent columns
    """
    logger.info(f"Creating sharing distribution plots...")

    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax1, ax2 = axes

    fig.suptitle(f'Sharing Distribution Analysis\n({stats["total_positions"]:,} positions, {num_parents} parents)',
                 fontsize=16, fontweight='bold')

    # Plot 1: Regular frequency histogram
    sharing_values = sorted(stats['sharing_distribution'].keys())
    sharing_frequencies = [stats['sharing_distribution'][x] for x in sharing_values]

    bars1 = ax1.bar(sharing_values, sharing_frequencies, alpha=0.7, color='skyblue',
                    edgecolor='black', width=0.8)

    ax1.set_xlabel('Number of Non-Zero Parents')
    ax1.set_ylabel('Frequency (Number of Positions)')
    ax1.set_title('Non-Zero Parent Count Distribution')
    ax1.grid(True, alpha=0.3)

    # Add percentage labels on bars
    for bar, sharing_count in zip(bars1, sharing_values):
        percentage = stats['sharing_percentages'][sharing_count]
        if percentage > 1:  # Only label bars with >1%
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)

    # Add statistics text
    stats_text = f'Mean: {stats["mean_sharing"]:.2f}\nMedian: {stats["median_sharing"]:.1f}\nStd: {stats["std_sharing"]:.2f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Count-scaled histogram
    if stats['scaled_distribution']:
        scaled_values = sorted(stats['scaled_distribution'].keys())
        scaled_frequencies = [stats['scaled_distribution'][x] for x in scaled_values]

        bars2 = ax2.bar(scaled_values, scaled_frequencies, alpha=0.7, color='lightcoral',
                        edgecolor='black', width=0.8)

        ax2.set_xlabel('Number of Non-Zero Parents')
        ax2.set_ylabel('Frequency (Scaled by Count Values)')
        ax2.set_title('Count-Scaled Distribution')
        ax2.grid(True, alpha=0.3)

        # Add percentage labels on bars for scaled plot
        for bar, sharing_count in zip(bars2, scaled_values):
            percentage = stats['scaled_percentages'][sharing_count]
            if percentage > 1:  # Only label bars with >1%
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                         f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)

        # Calculate scaled statistics
        scaled_mean = sum(k * v for k, v in stats['scaled_distribution'].items()) / stats['total_scaled']
        scaled_stats_text = f'Total Scaled: {stats["total_scaled"]:,}\nWeighted Mean: {scaled_mean:.2f}'
        ax2.text(0.02, 0.98, scaled_stats_text, transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No data for count-scaled plot\n(all positions have zero counts)',
                 ha='center', va='center', transform=ax2.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax2.set_xlabel('Number of Non-Zero Parents')
        ax2.set_ylabel('Frequency (Scaled by Count Values)')
        ax2.set_title('Count-Scaled Distribution')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Sharing distribution plots saved to {output_path}")

def check_zero_positions(data: np.ndarray) -> dict:
    """
    Check for positions where all features are zero.

    Args:
        data: Array of shape (n_windows, window_size, n_features)

    Returns:
        Dictionary with validation results
    """
    logger.info("=== CHECKING FOR ZERO POSITIONS ===")

    zero_positions = []

    for win_idx in range(data.shape[0]):
        for pos_idx in range(data.shape[1]):
            position_features = data[win_idx, pos_idx, :]

            if np.all(position_features == 0):
                zero_positions.append((win_idx, pos_idx))

    result = {
        'zero_positions_count': len(zero_positions),
        'zero_positions': zero_positions[:20],  # First 20 examples
        'passed': len(zero_positions) == 0
    }

    if len(zero_positions) > 0:
        logger.error(f"❌ Found {len(zero_positions)} positions with all-zero features!")
        for win_idx, pos_idx in zero_positions[:5]:
            logger.error(f"  Window {win_idx}, Position {pos_idx}: {data[win_idx, pos_idx, :]}")
    else:
        logger.info("✅ No positions with all-zero features found")

    return result


def normalize_diploid_labels(label1: int, label2: int) -> tuple:
    """
    Normalize diploid labels so that (0,1) and (1,0) are treated the same.
    Returns labels in sorted order.
    """
    return tuple(sorted([label1, label2]))


def check_label_consistency(data: np.ndarray, num_parents: int, window_sample_size: int = 100) -> dict:
    """
    Check label consistency within windows, accounting for diploid nature.

    Args:
        data: Array of shape (n_windows, window_size, n_features)
        num_parents: Number of parent columns
        window_sample_size: Number of windows to sample for detailed analysis

    Returns:
        Dictionary with validation results
    """
    logger.info("=== CHECKING LABEL CONSISTENCY ===")

    # Extract labels (last 2 columns)
    labels = data[:, :, num_parents:]  # Shape: (n_windows, window_size, 2)

    # Statistics
    total_windows = labels.shape[0]
    window_size = labels.shape[1]

    # Sample windows for detailed analysis
    sample_indices = np.random.choice(total_windows, min(window_sample_size, total_windows), replace=False)

    suspicious_windows = []
    label_change_stats = []

    for win_idx in sample_indices:
        window_labels = labels[win_idx]  # Shape: (window_size, 2)

        # Normalize diploid labels for comparison
        normalized_labels = [normalize_diploid_labels(int(row[0]), int(row[1])) for row in window_labels]

        # Count label changes
        changes = 0
        for i in range(1, len(normalized_labels)):
            if normalized_labels[i] != normalized_labels[i - 1]:
                changes += 1

        change_rate = changes / (window_size - 1) if window_size > 1 else 0
        label_change_stats.append(change_rate)

        # Flag windows with suspiciously high change rates
        if change_rate > 0.5:  # More than 50% positions have different labels than previous
            suspicious_windows.append({
                'window_idx': win_idx,
                'change_rate': change_rate,
                'changes': changes,
                'sample_labels': normalized_labels[:10]  # First 10 labels
            })

    # Overall statistics
    avg_change_rate = np.mean(label_change_stats)
    median_change_rate = np.median(label_change_stats)
    max_change_rate = np.max(label_change_stats)

    # Check for label value ranges
    all_label_values = np.unique(labels.flatten())

    result = {
        'total_windows_checked': len(sample_indices),
        'avg_change_rate': avg_change_rate,
        'median_change_rate': median_change_rate,
        'max_change_rate': max_change_rate,
        'suspicious_windows_count': len(suspicious_windows),
        'suspicious_windows': suspicious_windows[:10],  # First 10 examples
        'label_value_range': (int(all_label_values.min()), int(all_label_values.max())),
        'unique_label_values': sorted([int(x) for x in all_label_values]),
        'passed': len(suspicious_windows) < len(sample_indices) * 0.1  # Less than 10% suspicious
    }

    logger.info(f"Label consistency analysis on {len(sample_indices)} windows:")
    logger.info(f"  Average change rate: {avg_change_rate:.3f}")
    logger.info(f"  Median change rate: {median_change_rate:.3f}")
    logger.info(f"  Max change rate: {max_change_rate:.3f}")
    logger.info(f"  Label value range: {result['label_value_range']}")
    logger.info(f"  Unique labels: {result['unique_label_values']}")

    if len(suspicious_windows) > 0:
        logger.warning(f"⚠️  Found {len(suspicious_windows)} windows with high label change rates")
        for sw in suspicious_windows[:3]:
            logger.warning(f"  Window {sw['window_idx']}: {sw['changes']} changes, rate={sw['change_rate']:.3f}")
    else:
        logger.info("✅ No windows with suspicious label change patterns")

    return result


def check_label_runs(data: np.ndarray, num_parents: int, min_run_length: int = 5) -> dict:
    """
    Check that labels tend to appear in runs (not constantly switching).

    Args:
        data: Array of shape (n_windows, window_size, n_features)
        num_parents: Number of parent columns
        min_run_length: Minimum expected run length for good data

    Returns:
        Dictionary with validation results
    """
    logger.info("=== CHECKING LABEL RUNS ===")

    labels = data[:, :, num_parents:]  # Shape: (n_windows, window_size, 2)

    run_lengths = []
    short_run_windows = []

    # Sample some windows for analysis
    sample_size = min(1000, labels.shape[0])
    sample_indices = np.random.choice(labels.shape[0], sample_size, replace=False)

    for win_idx in sample_indices:
        window_labels = labels[win_idx]

        # Normalize diploid labels
        normalized_labels = [normalize_diploid_labels(int(row[0]), int(row[1])) for row in window_labels]

        # Find runs
        current_run_length = 1
        window_runs = []

        for i in range(1, len(normalized_labels)):
            if normalized_labels[i] == normalized_labels[i - 1]:
                current_run_length += 1
            else:
                window_runs.append(current_run_length)
                current_run_length = 1

        # Add the last run
        if current_run_length > 0:
            window_runs.append(current_run_length)

        run_lengths.extend(window_runs)

        # Check if this window has mostly short runs
        if window_runs and np.mean(window_runs) < min_run_length:
            short_run_windows.append({
                'window_idx': win_idx,
                'avg_run_length': np.mean(window_runs),
                'run_lengths': window_runs,
                'total_runs': len(window_runs)
            })

    if run_lengths:
        avg_run_length = np.mean(run_lengths)
        median_run_length = np.median(run_lengths)
        min_run = min(run_lengths)
        max_run = max(run_lengths)
    else:
        avg_run_length = median_run_length = min_run = max_run = 0

    result = {
        'total_runs_analyzed': len(run_lengths),
        'avg_run_length': avg_run_length,
        'median_run_length': median_run_length,
        'min_run_length': min_run,
        'max_run_length': max_run,
        'short_run_windows_count': len(short_run_windows),
        'short_run_windows': short_run_windows[:10],
        'passed': avg_run_length >= min_run_length and len(short_run_windows) < sample_size * 0.2
    }

    logger.info(f"Label run analysis:")
    logger.info(f"  Average run length: {avg_run_length:.2f}")
    logger.info(f"  Median run length: {median_run_length:.2f}")
    logger.info(f"  Run length range: [{min_run}, {max_run}]")

    if len(short_run_windows) > 0:
        logger.warning(f"⚠️  Found {len(short_run_windows)} windows with short label runs")
    else:
        logger.info("✅ Label runs appear normal")

    return result


def check_data_integrity(data: np.ndarray, num_parents: int) -> dict:
    """
    General data integrity checks.

    Args:
        data: Array of shape (n_windows, window_size, n_features)
        num_parents: Number of parent columns

    Returns:
        Dictionary with validation results
    """
    logger.info("=== CHECKING DATA INTEGRITY ===")

    results = {}

    # Basic shape validation
    expected_features = num_parents + 2
    if data.shape[2] != expected_features:
        logger.error(f"❌ Wrong number of features: expected {expected_features}, got {data.shape[2]}")
        results['shape_valid'] = False
    else:
        logger.info(f"✅ Correct shape: {data.shape}")
        results['shape_valid'] = True

    # Check for NaN or infinite values
    nan_count = np.sum(np.isnan(data))
    inf_count = np.sum(np.isinf(data))

    results['nan_count'] = nan_count
    results['inf_count'] = inf_count
    results['no_invalid_values'] = nan_count == 0 and inf_count == 0

    if nan_count > 0:
        logger.error(f"❌ Found {nan_count} NaN values")
    if inf_count > 0:
        logger.error(f"❌ Found {inf_count} infinite values")
    if nan_count == 0 and inf_count == 0:
        logger.info("✅ No NaN or infinite values found")

    # Check parent data ranges (should not be all the same)
    parent_data = data[:, :, :num_parents]
    parent_ranges = []

    for col in range(num_parents):
        col_data = parent_data[:, :, col].flatten()
        col_min, col_max = col_data.min(), col_data.max()
        col_range = col_max - col_min
        parent_ranges.append(col_range)

        if col_range == 0:
            logger.warning(f"⚠️  Parent column {col} has no variation (all values = {col_min})")

    results['parent_ranges'] = parent_ranges
    results['parents_have_variation'] = all(r > 0 for r in parent_ranges)

    # Check label data
    labels = data[:, :, num_parents:]
    label_min, label_max = labels.min(), labels.max()

    results['label_range'] = (float(label_min), float(label_max))
    results['labels_in_valid_range'] = label_min >= -1 and label_max <= num_parents

    logger.info(f"Parent data ranges: min={min(parent_ranges):.3f}, max={max(parent_ranges):.3f}")
    logger.info(f"Label range: [{label_min}, {label_max}]")

    if not results['labels_in_valid_range']:
        logger.error(f"❌ Labels outside expected range [0, {num_parents}]")
    else:
        logger.info("✅ Labels in valid range")

    return results


def generate_validation_report(data: np.ndarray, num_parents: int, output_path: Path = None):
    """
    Generate a comprehensive validation report.
    """
    logger.info("=== GENERATING VALIDATION REPORT ===")

    report = {
        'data_shape': data.shape,
        'num_parents': num_parents,
        'total_windows': data.shape[0],
        'window_size': data.shape[1],
        'total_features': data.shape[2]
    }

    # Run all checks
    logger.info("Running validation checks...")

    report['sharing_stats'] = analyze_sharing_distribution(data, num_parents, Path("sharing_distribution.png"))
    report['zero_positions'] = check_zero_positions(data)
    report['label_consistency'] = check_label_consistency(data, num_parents)
    report['label_runs'] = check_label_runs(data, num_parents)
    report['data_integrity'] = check_data_integrity(data, num_parents)

    # Overall pass/fail
    all_checks = [
        report['zero_positions']['passed'],
        report['label_consistency']['passed'],
        report['label_runs']['passed'],
        report['data_integrity']['shape_valid'],
        report['data_integrity']['no_invalid_values'],
        report['data_integrity']['parents_have_variation'],
        report['data_integrity']['labels_in_valid_range']
    ]

    report['overall_passed'] = all(all_checks)
    report['checks_passed'] = sum(all_checks)
    report['total_checks'] = len(all_checks)

    # Print summary
    logger.info("=== VALIDATION SUMMARY ===")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Checks passed: {report['checks_passed']}/{report['total_checks']}")

    logger.info(f"Mean sharing: {report['sharing_stats']['mean_sharing']:.2f}")
    logger.info(f"Distribution: {report['sharing_stats']['sharing_percentages']}")

    if report['overall_passed']:
        logger.info("🎉 ALL VALIDATION CHECKS PASSED!")
    else:
        logger.error("💥 SOME VALIDATION CHECKS FAILED!")

        # Detail failed checks
        if not report['zero_positions']['passed']:
            logger.error(f"  - Zero positions: {report['zero_positions']['zero_positions_count']} found")
        if not report['label_consistency']['passed']:
            logger.error(
                f"  - Label consistency: {report['label_consistency']['suspicious_windows_count']} suspicious windows")
        if not report['label_runs']['passed']:
            logger.error(f"  - Label runs: {report['label_runs']['short_run_windows_count']} windows with short runs")
        if not report['data_integrity']['shape_valid']:
            logger.error(f"  - Shape validation failed")
        if not report['data_integrity']['no_invalid_values']:
            logger.error(
                f"  - Invalid values: {report['data_integrity']['nan_count']} NaN, {report['data_integrity']['inf_count']} inf")
        if not report['data_integrity']['parents_have_variation']:
            logger.error(f"  - Some parent columns have no variation")
        if not report['data_integrity']['labels_in_valid_range']:
            logger.error(f"  - Labels outside valid range: {report['data_integrity']['label_range']}")

    # Save report if requested
    if output_path:
        logger.info(f"Saving detailed report to {output_path}")
        with open(output_path, 'w') as f:
            f.write("WINDOW EXTRACTION VALIDATION REPORT\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Data shape: {report['data_shape']}\n")
            f.write(f"Number of parents: {report['num_parents']}\n")
            f.write(f"Total windows: {report['total_windows']:,}\n")
            f.write(f"Window size: {report['window_size']}\n\n")

            f.write(f"OVERALL RESULT: {'PASS' if report['overall_passed'] else 'FAIL'}\n")
            f.write(f"Checks passed: {report['checks_passed']}/{report['total_checks']}\n\n")

            # Detailed results
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 20 + "\n")

            for check_name, check_result in report.items():
                if isinstance(check_result, dict) and 'passed' in check_result:
                    status = "PASS" if check_result['passed'] else "FAIL"
                    f.write(f"{check_name}: {status}\n")

                    # Add details for failed checks
                    if not check_result['passed']:
                        if 'zero_positions_count' in check_result:
                            f.write(f"  Zero positions found: {check_result['zero_positions_count']}\n")
                        if 'suspicious_windows_count' in check_result:
                            f.write(f"  Suspicious windows: {check_result['suspicious_windows_count']}\n")
                        if 'short_run_windows_count' in check_result:
                            f.write(f"  Short run windows: {check_result['short_run_windows_count']}\n")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate extracted window data for integrity and correctness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python validate_windows.py /path/to/windows.npy --num-parents 24

  # With detailed report
  python validate_windows.py /path/to/windows.npy --num-parents 24 --report-file validation_report.txt

  # Quick check only
  python validate_windows.py /path/to/windows.npy --num-parents 24 --quick
        """
    )

    parser.add_argument("input_file", help="Path to the .npy file to validate")
    parser.add_argument("--num-parents", type=int, required=True, help="Number of parent columns")
    parser.add_argument("--report-file", help="Save detailed report to this file")
    parser.add_argument("--quick", action="store_true", help="Run only essential checks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load data
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        return 1

    logger.info(f"Loading data from {input_path}...")
    try:
        data = np.load(input_path)
        logger.info(f"Loaded data shape: {data.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    # Validate data shape
    if data.ndim != 3:
        logger.error(f"Expected 3D data, got {data.ndim}D")
        return 1

    # Run validation
    if args.quick:
        logger.info("Running quick validation...")
        zero_check = check_zero_positions(data)
        integrity_check = check_data_integrity(data, args.num_parents)

        if zero_check['passed'] and integrity_check['shape_valid'] and integrity_check['no_invalid_values']:
            logger.info("✅ Quick validation passed!")
            return 0
        else:
            logger.error("❌ Quick validation failed!")
            return 1
    else:
        # Full validation
        report_path = Path(args.report_file) if args.report_file else None
        report = generate_validation_report(data, args.num_parents, report_path)

        return 0 if report['overall_passed'] else 1


if __name__ == "__main__":
    exit(main())