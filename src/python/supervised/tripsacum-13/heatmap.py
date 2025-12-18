import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd


# Create heatmap data
sample = "M162W_1_Oh7B_1"
chr = 2

data = np.load(f"../../cross/diploid-data/validation/{sample}_chr{chr}_matrix.npy", allow_pickle=True)
print(data.shape)

matrix = data[:, :-2]
all_labels_1 = data[:, -1].astype(float)
all_labels_1[all_labels_1 == -1] = np.nan
all_labels_2 = data[:, -2].astype(float)
all_labels_2[all_labels_2 == -1] = np.nan
background = matrix > 0


# all_predictions = np.load(f"../../cross/model_preds/diploid-collapsed/{sample}_chr{chr}.npy", allow_pickle=True).astype(float)
# print(all_predictions.shape)
# all_predictions_1 = all_predictions[:, :, 0].flatten()
# print(all_predictions_1.shape)
# all_predictions_2 = all_predictions[:, :, 1].flatten()
# print(all_predictions_2.shape)

start = 0
end = 100000

background_limited = background[start:end]  # Adjust shape as needed
true_labels_1 = all_labels_1[start:end]
true_labels_2 = all_labels_2[start:end]

# Transpose the heatmap data
background_limited = background_limited.T  # Transpose the background data

# Plot background heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    background_limited,
    cmap= LinearSegmentedColormap.from_list("GrayWhite", ["white", "gray"]),
    cbar=False,
    yticklabels=True,
    xticklabels=False  # Enable tick labels for x-axis
)

### UNCOMMENT IF YOU HAVE IMPUTED MODEL PREDICTIONS

# all_predictions[all_predictions > 12] = np.nan
# print("path 1 chromosome accuracy: ", (all_labels_1[:len(all_predictions_1)] == all_predictions_1).mean() + (all_labels_2[:len(all_predictions_1)] == all_predictions_1).mean())
# print("path 2 chromosome accuracy: ", (all_labels_2[:len(all_predictions_2)] == all_predictions_2).mean() + (all_labels_1[:len(all_predictions_2)] == all_predictions_2).mean())
#
#
# matched_labels_all_1 = np.where(
#     all_labels_1[:len(all_predictions_1)] == all_predictions_1,
#     all_labels_1[:len(all_predictions_1)],
#     np.nan
# )
# predictions_1 = all_predictions_1[start:end]
# matched_labels_1 = matched_labels_all_1[start:end]
# print("path 1 window accuracy: ", (true_labels_1 == predictions_1).mean() + (true_labels_2 == predictions_1).mean())
#
# matched_labels_all_2 = np.where(
#     all_labels_2[:len(all_predictions_2)] == all_predictions_2,
#     all_labels_2[:len(all_predictions_2)],
#     np.nan
# )
# predictions_2 = all_predictions_2[start:end]
# matched_labels_2 = matched_labels_all_2[start:end]
# print("path 2 window accuracy: ", (true_labels_2 == predictions_2).mean() + (true_labels_1 == predictions_2).mean())
#
# # Overlay true labels as another line
# plt.plot(
#     range(len(true_labels_1)),  # y-coordinates (sample index after switching)
#     true_labels_1 + 0.5,                  # x-coordinates (switched to align with transposed heatmap)
#     color="black",                  # Line color for true labels
#     label="Path 1",
#     linewidth=5                   # Line thickness
# )
#
# # Overlay predictions_bestpt as another line
# plt.plot(
#     range(len(true_labels_2)),  # y-coordinates (sample index after switching)
#     true_labels_2 + 0.5,                  # x-coordinates (switched to align with transposed heatmap)
#     color="black",                  # Line color for predicted labels
#     label="Path 2",
#     linewidth=5                   # Line thickness
# )

# # Overlay predictions 1
# plt.plot(
#     range(len(predictions_1)),  # y-coordinates (sample index after switching)
#     predictions_1 + 0.5,                  # x-coordinates (switched to align with transposed heatmap)
#     color="cyan",                  # Line color for predicted labels
#     label="Predictions 1",
#     linewidth=2                   # Line thickness
# )
# # Overlay predictions 2
# plt.plot(
#     range(len(predictions_2)),  # y-coordinates (sample index after switching)
#     predictions_2 + 0.5,                  # x-coordinates (switched to align with transposed heatmap)
#     color="magenta",                  # Line color for predicted labels
#     label="Predictions 2",
#     linewidth=2                   # Line thickness
# )
#
#
# # Overlay correct predictions 1
# plt.plot(
#     range(len(predictions_1)),  # y-coordinates (sample index after switching)
#     matched_labels_1 + 0.5,                  # x-coordinates (switched to align with transposed heatmap)
#     color="cyan",                  # Line color for predicted labels
#     label="Correct Predictions 1",
#     linewidth=2                   # Line thickness
# )
# # Overlay correct predictions 2
# plt.plot(
#     range(len(predictions_2)),  # y-coordinates (sample index after switching)
#     matched_labels_2 + 0.5,                  # x-coordinates (switched to align with transposed heatmap)
#     color="magenta",                  # Line color for predicted labels
#     label="Correct Predictions 2",
#     linewidth=2                   # Line thickness
# )

#Add legend
#plt.legend(loc="upper right", fontsize=12)

########################################################

# Title and labels
plt.title(f"{sample} chr{chr} {start} to {end}", fontsize=16)
plt.xlabel("Label")
plt.ylabel("Feature Index")
plt.show()