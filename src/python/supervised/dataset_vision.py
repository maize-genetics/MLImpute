# File to hold the various iterations of datasets for this training
# "Default" is segmentation dataset
# also holds the data collator for this dataset
from transformers.data.data_collator import InputDataClass
from typing import Any
from transformers.data.data_collator import default_data_collator
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd


# Slightly modified from default data collator
# Pads input labels and attention masks to the length of the longest sequence in the batch
# assumes that features have keys: "labels", "decoder_input_ids", and "decoder_attention_mask"
def custom_data_collator(features: list[InputDataClass]) -> dict[str, Any]:
    # loss calculations will ignore this token
    padding_token = -100
    input_padding_token = 6144 # TODO remove hard-coding

    # pad label features to the length of the longest
    # right-padded for training
    longest_seq = np.max([feat["labels"].shape[0] for feat in features])

    for feat in features:
        if feat["labels"].shape[0] < longest_seq:
            pad_len = longest_seq-feat["labels"].shape[0]
            feat["labels"] = np.concatenate((feat["labels"], [padding_token]*pad_len))
            feat["decoder_input_ids"] = np.concatenate((feat["decoder_input_ids"],
                                                        [input_padding_token]*pad_len))
            feat["decoder_attention_mask"] = np.concatenate((feat["decoder_attention_mask"],
                                                             np.zeros(pad_len)))

        feat["labels"] = torch.tensor(feat["labels"], dtype=torch.int64).clone()
        feat["decoder_input_ids"] = torch.tensor(feat["decoder_input_ids"], dtype=torch.int64).clone()

    # pass padded tokens to default collator
    batch = default_data_collator(features)
    # batch["return_dict"] = False

    return batch

# The "default" dataset object
# We use a list of npy files to store read counts from PS4G files and randomly access them with mmap
# to make the data suitable for input to the vision model we wrap it to a square format,
# normalize it, and create the three input channels
class SegmentationDataset(Dataset):
    def __init__(self, keyfile, image_size=384, num_parents=24, step_size=1536, windows=None, split_norm_levels=False, include_index=False):
        self.keyfile = pd.read_csv(keyfile, sep="\t")
        self.window_size = (image_size * image_size) // num_parents
        self.image_size = image_size
        self.num_parents = num_parents
        self.step_size = step_size
        if windows is not None:
            self.windows = list(pd.read_csv(windows, sep="\t").itertuples(index=False))
        else:
            self.windows = self.__generate_windows__()
        self.split_norm_levels = split_norm_levels
        self.include_index = include_index

        self.n_windows = len(self.windows)

    # uses a list to store all valid windows
    # this could be manipulated to skip windows with unlabeled bins
    def __generate_windows__(self):
        # windows is a list of tuples, where each tuple represents a training data point
        # the tuples are formatted as (file index, window step index)
        # multiply window step index by step_size to get the index of the first position in the window
        windows = [] # file idx, window step idx

        for idx in range(len(self.keyfile)):
            filelen = self.keyfile.iloc[idx]["length"]
            num_windows = (filelen - self.window_size) // self.step_size
            windows.extend([(idx, idy) for idy in range(num_windows)])

        return windows

    def __len__(self):
        return self.n_windows

    # converts per_position labels to a sequence of transition points
    # this is needed because labels are stored on a per-position basis,
    # but we want to train on the location of each crossover point (relative to the context window)
    def __bins_to_idx__(self, labels_binned):
        return [idx+1 for idx in range(labels_binned.shape[0] - 1) if labels_binned[idx] != labels_binned[idx+1]]

    def __getitem__(self, idx):
        # retrieve window index from list
        file_idx, pos_idx = self.windows[idx]

        # convert to position start and end
        pos_start = pos_idx * self.step_size
        pos_end = pos_start + self.window_size

        # grab segment from mmaped numpy
        ip = np.load(self.keyfile["path"].iloc[file_idx], allow_pickle=True, mmap_mode='r')[pos_start:pos_end]

        # separate labels and generate junctions aka crossover points
        labels_binned = ip[:, self.num_parents]
        junctions = self.__bins_to_idx__(labels_binned)

        matrix = ip[:, 0:self.num_parents]
        matrix = np.hstack(np.split(matrix, self.window_size // self.image_size))

        # normalize the matrix according to ViT's requirements (mean 0.5 std 0.5)
        mean = np.mean(matrix)
        sd = np.std(matrix)
        matrix_r = (matrix - mean) / sd
        matrix_r = matrix_r * 0.5 + 0.5

        if self.split_norm_levels:
            matrix_g = (matrix - self.keyfile.iloc[file_idx]["chrom_mean"]) / self.keyfile.iloc[file_idx]["chrom_stdev"]
            matrix_g = matrix_g * 0.5 + 0.5

            matrix_b = (matrix - self.keyfile.iloc[file_idx]["global_mean"]) / self.keyfile.iloc[file_idx]["global_stdev"]
            matrix_b = matrix_b * 0.5 + 0.5

            window = np.stack((matrix_r, matrix_g, matrix_b), axis=0)

        else:
            # wrap input data to the square aspect ratio
            # and triple it to produce a greyscale RGB image
            window = np.stack((matrix_r, matrix_r, matrix_r), axis=0)

        # We've got labels and decoder input id's separately here, even though they should just be
        # shifted versions of one another. This was part of debugging that I did because
        # I think the default shifting function built into the model isn't working right
        # for our needs. Specifying both overrides the default though.
        labels = np.concatenate((junctions, [self.window_size + 1]))
        input_ids = np.concatenate(([self.window_size+2], junctions))
        mask = np.ones(len(junctions) + 1)

        # Note: we are relying on the data collator to handle padding, so labels do not have
        # a fixed length
        if self.include_index:
            return {
                'pixel_values': torch.tensor(window, dtype=torch.float),  #(3, image_size, image_size)
                'labels': labels,  # (torch.int64, variable length)
                'decoder_attention_mask': mask,  # (boolean, same length as labels)
                'decoder_input_ids': torch.tensor(input_ids),  # (torch.int64, same length as labels)
                'file_idx': file_idx,
                'pos_idx': pos_idx
            }
        else:
            return {
                'pixel_values': torch.tensor(window, dtype=torch.float),  #(3, image_size, image_size)
                'labels': labels,  # (torch.int64, variable length)
                'decoder_attention_mask': mask,  # (boolean, same length as labels)
                'decoder_input_ids': torch.tensor(input_ids)  # (torch.int64, same length as labels)
            }

