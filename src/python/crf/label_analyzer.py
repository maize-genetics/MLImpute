import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import argparse
import os
from pathlib import Path
import json


class LabelChangeAnalyzer:
    def __init__(self):
        self.stats = {}

    def analyze_label_changes(self, data):
        """
        Analyze label changes in a numpy array where the last column contains labels.

        Parameters:
        data: numpy array of shape [position, features] where last column is labels

        Returns:
        Dictionary containing statistics about label changes
        """
        # Extract labels (last column)
        labels = data[:, -1]

        # Find where labels change
        label_changes = np.where(np.diff(labels) != 0)[0]

        # Total number of switches (changes between different labels)
        total_switches = len(label_changes)

        # Calculate run lengths
        if len(label_changes) == 0:
            # No changes - entire sequence is one label
            run_lengths = [len(labels)]
            unique_labels = [labels[0]]
        else:
            # Add start and end positions for easier calculation
            change_positions = np.concatenate(([0], label_changes + 1, [len(labels)]))
            run_lengths = np.diff(change_positions)

            # Get the label for each run
            run_start_indices = change_positions[:-1]
            unique_labels = labels[run_start_indices]

        # Calculate statistics
        avg_run_length = np.mean(run_lengths)
        median_run_length = np.median(run_lengths)
        min_run_length = np.min(run_lengths)
        max_run_length = np.max(run_lengths)
        std_run_length = np.std(run_lengths)

        # Count occurrences of each label
        label_counts = Counter(labels)

        # Count runs of each label
        run_label_counts = Counter(unique_labels)

        self.stats = {
            'total_positions': len(labels),
            'total_switches': total_switches,
            'num_runs': len(run_lengths),
            'avg_run_length': avg_run_length,
            'median_run_length': median_run_length,
            'min_run_length': min_run_length,
            'max_run_length': max_run_length,
            'std_run_length': std_run_length,
            'run_lengths': run_lengths.tolist(),  # Convert to list for JSON serialization
            'label_counts': dict(label_counts),
            'run_label_counts': dict(run_label_counts),
            'unique_labels': sorted(list(label_counts.keys())),
            'switch_rate': total_switches / len(labels) if len(labels) > 0 else 0
        }

        return self.stats

    def print_stats(self, verbose=True):
        """Print formatted statistics"""
        if not self.stats:
            print("No analysis has been run yet!")
            return

        if verbose:
            print("=" * 50)
            print("LABEL CHANGE ANALYSIS")
            print("=" * 50)

        print(f"Total positions: {self.stats['total_positions']}")
        print(f"Total label switches: {self.stats['total_switches']}")
        print(f"Number of runs: {self.stats['num_runs']}")
        print(f"Switch rate: {self.stats['switch_rate']:.4f}")

        if verbose:
            print()
            print("RUN LENGTH STATISTICS:")
            print(f"  Average run length: {self.stats['avg_run_length']:.2f}")
            print(f"  Median run length: {self.stats['median_run_length']:.2f}")
            print(f"  Std run length: {self.stats['std_run_length']:.2f}")
            print(f"  Min run length: {self.stats['min_run_length']}")
            print(f"  Max run length: {self.stats['max_run_length']}")
            print()

            print("LABEL DISTRIBUTION:")
            for label in sorted(self.stats['label_counts'].keys()):
                count = self.stats['label_counts'][label]
                percentage = (count / self.stats['total_positions']) * 100
                print(f"  Label {label}: {count} positions ({percentage:.1f}%)")
            print()

            print("RUN DISTRIBUTION BY LABEL:")
            for label in sorted(self.stats['run_label_counts'].keys()):
                count = self.stats['run_label_counts'][label]
                print(f"  Label {label}: {count} runs")

    def plot_analysis(self, data, output_dir=None, filename_prefix="", show_plots=True, save_plots=False):
        """Create visualizations of the label analysis"""
        if not self.stats:
            print("No analysis has been run yet!")
            return

        labels = data[:, -1]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Labels over positions
        ax1.plot(labels, 'b-', alpha=0.7, linewidth=1)
        ax1.set_title('Labels Over Position')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Label')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Histogram of run lengths
        ax2.hist(self.stats['run_lengths'], bins=min(20, len(self.stats['run_lengths'])),
                 alpha=0.7, color='green', edgecolor='black')
        ax2.set_title('Distribution of Run Lengths')
        ax2.set_xlabel('Run Length')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Label counts
        labels_list = sorted(self.stats['label_counts'].keys())
        counts_list = [self.stats['label_counts'][label] for label in labels_list]
        ax3.bar(range(len(labels_list)), counts_list, alpha=0.7, color='orange')
        ax3.set_title('Total Count by Label')
        ax3.set_xlabel('Label')
        ax3.set_ylabel('Count')
        ax3.set_xticks(range(len(labels_list)))
        ax3.set_xticklabels(labels_list)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Run counts by label
        run_labels = sorted(self.stats['run_label_counts'].keys())
        run_counts = [self.stats['run_label_counts'][label] for label in run_labels]
        ax4.bar(range(len(run_labels)), run_counts, alpha=0.7, color='red')
        ax4.set_title('Number of Runs by Label')
        ax4.set_xlabel('Label')
        ax4.set_ylabel('Number of Runs')
        ax4.set_xticks(range(len(run_labels)))
        ax4.set_xticklabels(run_labels)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plot_filename = f"{filename_prefix}label_analysis.png"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")

        if show_plots:
            plt.show()
        else:
            plt.close()

    def save_stats_to_json(self, output_dir, filename_prefix=""):
        """Save statistics to JSON file"""
        if not self.stats:
            print("No analysis has been run yet!")
            return None

        os.makedirs(output_dir, exist_ok=True)
        json_filename = f"{filename_prefix}label_stats.json"
        json_path = os.path.join(output_dir, json_filename)

        with open(json_path, 'w') as f:
            json.dump(self.stats, f, indent=2)

        print(f"Statistics saved to: {json_path}")
        return json_path


def load_data(file_path):
    """Load data from various file formats"""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix == '.npy':
        return np.load(file_path)
    elif file_path.suffix == '.npz':
        npz_file = np.load(file_path)
        # Assume the first array in the npz file is the data
        return npz_file[list(npz_file.keys())[0]]
    elif file_path.suffix in ['.txt', '.csv']:
        return np.loadtxt(file_path, delimiter=',')
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def run_analysis(input_file, output_dir="results", filename_prefix="",
                 verbose=True, save_json=True, save_plots=False, show_plots=True):
    """
    Run the complete label change analysis

    Parameters:
    input_file: Path to input data file
    output_dir: Directory to save results
    filename_prefix: Prefix for output filenames
    verbose: Whether to print detailed statistics
    save_json: Whether to save statistics to JSON
    save_plots: Whether to save plots to files
    show_plots: Whether to display plots
    """

    print(f"Loading data from: {input_file}")
    data = load_data(input_file)

    if not filename_prefix:
        filename_prefix = f"{Path(input_file).stem}_"

    print(f"Data shape: {data.shape}")

    # Validate data format
    if len(data.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape: {data.shape}")

    # Run analysis
    analyzer = LabelChangeAnalyzer()
    stats = analyzer.analyze_label_changes(data)

    # Print results
    analyzer.print_stats(verbose=verbose)

    # Save results
    if save_json:
        analyzer.save_stats_to_json(output_dir, filename_prefix)

    # Create plots
    analyzer.plot_analysis(data, output_dir, filename_prefix, show_plots, save_plots)

    return analyzer, data


def main():
    parser = argparse.ArgumentParser(description='Analyze label changes in numpy arrays')
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--output-dir', '-o', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--prefix', '-p', type=str, default='',
                        help='Prefix for output filenames')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Less verbose output')
    parser.add_argument('--no-json', action='store_true',
                        help='Don\'t save statistics to JSON')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to files')
    parser.add_argument('--no-show-plots', action='store_true',
                        help='Don\'t display plots')

    args = parser.parse_args()

    try:
        run_analysis(
            input_file=args.input,
            output_dir=args.output_dir,
            filename_prefix=args.prefix,
            verbose=not args.quiet,
            save_json=not args.no_json,
            save_plots=args.save_plots,
            show_plots=not args.no_show_plots
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


# Example usage functions for batch processing
def analyze_multiple_files(file_list, output_base_dir="batch_results",
                           save_json=True, save_plots=True, show_plots=False, verbose=False):
    """
    Analyze multiple files in batch

    Parameters:
    file_list: List of file paths to analyze
    output_base_dir: Base directory for all results
    save_json: Whether to save JSON stats for each file
    save_plots: Whether to save plots for each file
    show_plots: Whether to display plots (usually False for batch)
    verbose: Whether to print detailed stats for each file
    """
    results = {}

    for file_path in file_list:
        try:
            print(f"\n{'=' * 60}")
            print(f"ANALYZING: {file_path}")
            print(f"{'=' * 60}")

            file_stem = Path(file_path).stem
            output_dir = os.path.join(output_base_dir, file_stem)

            analyzer, data = run_analysis(
                input_file=file_path,
                output_dir=output_dir,
                filename_prefix=f"{file_stem}_",
                verbose=verbose,
                save_json=save_json,
                save_plots=save_plots,
                show_plots=show_plots
            )

            results[file_path] = analyzer.stats

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            results[file_path] = None

    # Save combined results
    if save_json:
        os.makedirs(output_base_dir, exist_ok=True)
        combined_results_path = os.path.join(output_base_dir, "combined_results.json")
        with open(combined_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nCombined results saved to: {combined_results_path}")

    return results


def analyze_directory(directory_path, pattern="*.npy", **kwargs):
    """
    Analyze all files matching a pattern in a directory

    Parameters:
    directory_path: Path to directory containing files
    pattern: Glob pattern for files to analyze (default: "*.npy")
    **kwargs: Additional arguments passed to analyze_multiple_files
    """
    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = list(directory.glob(pattern))

    if not files:
        print(f"No files found matching pattern '{pattern}' in {directory}")
        return {}

    print(f"Found {len(files)} files to analyze:")
    for f in files:
        print(f"  - {f}")

    return analyze_multiple_files([str(f) for f in files], **kwargs)


if __name__ == "__main__":
    exit(main())