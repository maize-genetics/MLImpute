import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, help="sample name")
    parser.add_argument("--chr", type=str, help="chromosome/contig name")
    parser.add_argument("--start", type=int, help="start position")
    parser.add_argument("--end", type=int, help="end position")
    parser.add_argument("--matrix-dir", type=str, help="path to matrix directory")
    parser.add_argument("--predictions", type=str, help="path to predictions numpy file")
    parser.add_argument("--diploid", action="store_true", help="add this flag for diploid predictions")
    args = parser.parse_args()

    # Load in matrix
    sample = args.sample
    chr = args.chr

    data = np.load(f"{args.matrix_dir}/{sample}_{chr}_matrix.npy", allow_pickle=True)

    all_labels_1 = data[:, -1].astype(float)
    all_labels_1[all_labels_1 == -1] = np.nan

    if args.diploid:
        assert(data.shape[1] == 26)
        matrix = data[:, :-2]
        all_labels_2 = data[:, -2].astype(float)
        all_labels_2[all_labels_2 == -1] = np.nan
    else:
        assert(data.shape[1] == 25)
        matrix = data[:, :-1]

    assert(matrix.min() >= 0)
    assert(matrix.max() <= 127)

    # load in predictions
    all_predictions = np.load(args.predictions, allow_pickle=True).astype(float)
    all_predictions[all_predictions >= matrix.shape[1]] = np.nan

    if args.diploid:
        assert (all_predictions.shape[-1] == 2)
        all_predictions = all_predictions.reshape(-1, all_predictions.shape[-1])
        all_predictions_1 = all_predictions[:, 0].flatten()
        all_predictions_2 = all_predictions[:, 1].flatten()
    else:
        all_predictions_1 = all_predictions.flatten()

    # build window
    start = args.start
    end = args.end
    assert(start < end)

    window_matrix = matrix[start:end].T
    true_labels_1 = all_labels_1[start:end]
    predictions_1 = all_predictions_1[start:end]

    if args.diploid:
        true_labels_2 = all_labels_2[start:end]
        predictions_2 = all_predictions_2[start:end]
        print("path 1 chromosome accuracy: ", (all_labels_1[:len(all_predictions_1)] == all_predictions_1).mean() + (all_labels_2[:len(all_predictions_1)] == all_predictions_1).mean())
        print("path 2 chromosome accuracy: ", (all_labels_2[:len(all_predictions_2)] == all_predictions_2).mean() + (all_labels_1[:len(all_predictions_2)] == all_predictions_2).mean())
        print("path 1 window accuracy: ", (true_labels_1 == predictions_1).mean() + (true_labels_2 == predictions_1).mean())
        print("path 2 window accuracy: ", (true_labels_2 == predictions_2).mean() + (true_labels_1 == predictions_2).mean())
    else:
        print("path 1 chromosome accuracy: ", (all_labels_1[:len(all_predictions_1)] == all_predictions_1).mean())
        print("path 1 window accuracy: ", (true_labels_1 == predictions_1).mean())


    # Plot background heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        window_matrix,
        cmap="Greys",
        norm=PowerNorm(gamma=0.1, vmin=0, vmax=127),
        cbar=True,
        yticklabels=True,
        xticklabels=False
    )

    # Overlay true labels 1
    plt.plot(
        range(len(true_labels_1)),  # y-coordinates (sample index after switching)
        true_labels_1 + 0.5,                  # x-coordinates (switched to align with transposed heatmap)
        color="black",                  # Line color for true labels
        label="Path 1",
        linewidth=5                   # Line thickness
    )

    if args.diploid:
        # Overlay true labels 2
        plt.plot(
            range(len(true_labels_2)),  # y-coordinates (sample index after switching)
            true_labels_2 + 0.5,  # x-coordinates (switched to align with transposed heatmap)
            color="black",  # Line color for true labels
            label="Path 2",
            linewidth=5  # Line thickness
        )

    # Overlay predictions 1
    plt.plot(
        range(len(predictions_1)),  # y-coordinates (sample index after switching)
        predictions_1 + 0.5,                  # x-coordinates (switched to align with transposed heatmap)
        color="cyan",                  # Line color for predicted labels
        label="Predictions 1",
        linewidth=2                   # Line thickness
    )

    if args.diploid:
        # Overlay predictions 2
        plt.plot(
            range(len(predictions_2)),  # y-coordinates (sample index after switching)
            predictions_2 + 0.5,  # x-coordinates (switched to align with transposed heatmap)
            color="magenta",  # Line color for predicted labels
            label="Predictions 2",
            linewidth=2  # Line thickness
        )

    # Add legend
    plt.legend(loc="upper right", fontsize=12)

    # Title and labels
    plt.title(f"{sample} {chr} {start} to {end}", fontsize=16)
    plt.xlabel("Label")
    plt.ylabel("Feature Index")
    plt.show()

if __name__ == "__main__":
    main()