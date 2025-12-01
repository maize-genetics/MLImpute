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
from scipy.stats import binom


# Slightly modified from default data collator
# Pads input labels and attention masks to the length of the longest sequence in the batch
# assumes that features have keys: "labels", "decoder_input_ids", and "decoder_attention_mask"
def custom_data_collator(features: list[InputDataClass]) -> dict[str, Any]:
    # loss calculations will ignore this token
    padding_token = -100
    input_padding_token = 6144  # TODO remove hard-coding

    # pad label features to the length of the longest
    # right-padded for training
    longest_seq = np.max([feat["labels"].shape[0] for feat in features])

    for feat in features:
        if feat["labels"].shape[0] < longest_seq:
            pad_len = longest_seq - feat["labels"].shape[0]
            # feat["labels"] = np.concatenate((feat["labels"], [padding_token]*pad_len))
            feat["decoder_input_ids"] = np.concatenate((feat["decoder_input_ids"],
                                                        [input_padding_token] * pad_len))
            feat["decoder_attention_mask"] = np.concatenate((feat["decoder_attention_mask"],
                                                             np.zeros(pad_len)))

            # feat["labels_smoothed"] = np.concatenate((
            #     feat["labels_smoothed"], np.full(
            #         (pad_len,
            #          feat["labels_smoothed"].shape[1],
            #          feat["labels_smoothed"].shape[2]), -1)))

            feat["labels"] = np.concatenate((
                feat["labels"], np.full((pad_len, feat["labels"].shape[1]), 10)
            ))

        # feat["labels"] = torch.tensor(feat["labels"], dtype=torch.int64).clone()
        # feat["labels_smoothed"] = torch.tensor(feat["labels_smoothed"], dtype=torch.int64).clone()
        feat["labels"] = torch.tensor(feat["labels"]).clone()
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
    def __init__(self, keyfile, image_size=384, num_parents=24, step_size=1536, windows=None, split_norm_levels=False,
                 include_index=False):
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
        windows = []  # file idx, window step idx

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
        return [idx + 1 for idx in range(labels_binned.shape[0] - 1) if labels_binned[idx] != labels_binned[idx + 1]]

    def fit_edge(self, idx):
        if idx < 0:
            return 0
        elif idx > self.window_size:
            return self.window_size
        else:
            return idx

    def spread(self, idx):
        x = torch.zeros((self.window_size, 4))
        x[idx, 0] = 1
        x[self.fit_edge(idx - 4): self.fit_edge(idx + 5), 1] = 1
        x[self.fit_edge(idx - 16): self.fit_edge(idx + 17), 2] = 1
        x[self.fit_edge(idx - 32): self.fit_edge(idx + 33), 3] = 1

        return x

    def dist(self, idx):
        x = torch.zeros((self.window_size))
        # TODO: parameterize
        midpoint = 32 // 2

        for idy in range(32):
            if 0 <= idx + idy - midpoint < self.window_size:
                x[idx + idy - midpoint] = binom.pmf(idy, 32, 0.5)

        return x

    def __distribute_labels__(self, labels_binned):
        return [self.spread(idx + 1) for idx in range(labels_binned.shape[0] - 1) if
                labels_binned[idx] != labels_binned[idx + 1]]

    def __distribute_labels_2__(self, labels_binned):
        return [self.dist(idx + 1) for idx in range(labels_binned.shape[0] - 1) if
                labels_binned[idx] != labels_binned[idx + 1]]

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

            matrix_b = (matrix - self.keyfile.iloc[file_idx]["global_mean"]) / self.keyfile.iloc[file_idx][
                "global_stdev"]
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
        input_ids = np.concatenate(([self.window_size + 2], junctions))
        labels_binom = np.array(self.__distribute_labels_2__(ip[:, self.num_parents]))
        labels0 = np.concatenate((junctions, [self.window_size + 1]))

        # add special tokens and end token
        if labels_binom.shape[0] == 0:  # no crossovers

            labels_binom = np.zeros((1, self.window_size + 3))
        else:  # yes crossovers

            labels_binom = np.concatenate((labels_binom, np.zeros((labels_binom.shape[0], 3))), axis=1)
            labels_binom = np.concatenate((labels_binom, np.zeros((1, labels_binom.shape[1]))), axis=0)

        # add end token label (no smoothing)
        labels_binom[labels_binom.shape[0] - 1, self.window_size + 1] = 1

        mask = np.ones(len(junctions) + 1)

        # Note: we are relying on the data collator to handle padding, so labels do not have
        # a fixed length
        if self.include_index:
            return {
                'pixel_values': torch.tensor(window, dtype=torch.float),  # (3, image_size, image_size)
                "labels": torch.tensor(labels_binom),
                "int_labels": torch.tensor(labels0),
                'decoder_attention_mask': mask,  # (boolean, same length as labels)
                'decoder_input_ids': torch.tensor(input_ids),  # (torch.int64, same length as labels)
                'file_idx': file_idx,
                'pos_idx': pos_idx
            }
        else:
            return {
                'pixel_values': torch.tensor(window, dtype=torch.float),  # (3, image_size, image_size)
                "labels": labels_binom,
                'decoder_attention_mask': mask,  # (boolean, same length as labels)
                'decoder_input_ids': torch.tensor(input_ids)  # (torch.int64, same length as labels)
            }


class CategoricalSegmentationDataset(SegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

            matrix_b = (matrix - self.keyfile.iloc[file_idx]["global_mean"]) / self.keyfile.iloc[file_idx][
                "global_stdev"]
            matrix_b = matrix_b * 0.5 + 0.5

            window = np.stack((matrix_r, matrix_g, matrix_b), axis=0)

        else:
            # wrap input data to the square aspect ratio
            # and triple it to produce a greyscale RGB image
            window = np.stack((matrix_r, matrix_r, matrix_r), axis=0)

        if len(junctions) > 0:
            label = 1
        else:
            label = 0

        # Note: we are relying on the data collator to handle padding, so labels do not have
        # a fixed length
        if self.include_index:
            return {
                'pixel_values': torch.tensor(window, dtype=torch.float),  # (3, image_size, image_size)
                "labels": label,
                'file_idx': file_idx,
                'pos_idx': pos_idx
            }
        else:
            return {
                'pixel_values': torch.tensor(window, dtype=torch.float),  # (3, image_size, image_size)
                "labels": label
            }

class CategoricalBertDataset(SegmentationDataset):
    def __init__(self, keyfile, seq_len=64, num_parents=24, step_size=16, windows=None, include_index=False):
        self.keyfile = pd.read_csv(keyfile, sep="\t")
        self.window_size = seq_len
        self.num_parents = num_parents
        self.step_size = step_size
        if windows is not None:
            self.windows = list(pd.read_csv(windows, sep="\t").itertuples(index=False))
        else:
            self.windows = self.__generate_windows__()
        self.include_index = include_index

        self.n_windows = len(self.windows)

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

        # normalize the matrix according to ViT's requirements (mean 0.5 std 0.5)
        mean = np.mean(matrix)
        sd = np.std(matrix)
        matrix = (matrix - mean) / sd
        matrix = matrix * 0.5 + 0.5

        if len(junctions) > 0:
            label = 1
        else:
            label = 0

        # Note: we are relying on the data collator to handle padding, so labels do not have
        # a fixed length
        if self.include_index:
            return {
                'inputs_embeds': torch.tensor(matrix, dtype=torch.float),
                "labels": label,
                'file_idx': file_idx,
                'pos_idx': pos_idx
            }
        else:
            return {
                'inputs_embeds': torch.tensor(matrix, dtype=torch.float),
                "labels": label
            }
class SumCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, logits, labels, vocab_size=None, num_items_in_batch=None):
        return self.loss(torch.permute(logits, (0, 2, 1)), labels)


# for SOME REASON, ViT decided to reverse the order of input and target relative to every other torch loss function
# so we can't just pass them in unchanged, we have to put a wrapper over them
# and no, we didn't even use keyword arguments to help clear up any confusion
class ClassifierCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, labels, logits, vocab_size=None, num_items_in_batch=None):
        return self.loss(logits, labels)

class BinnedCrossEntropy(torch.nn.Module):
    def __init__(self, spread, max_token, reduction="mean"):
        super().__init__()
        self.spread = spread
        self.max_token = max_token
        self.reduction = reduction

    def forward(self, logits, labels, vocab_size=None, num_items_in_batch=None):
        y_pred = torch.softmax(logits, dim=2)

        loss = 0  # torch.zeros((y_pred.shape[0], y_pred.shape[1]))

        for idx in range(y_pred.shape[0]):
            for idy in range(y_pred.shape[1]):
                if labels[idx, idy] >= 0:  # ignore -100 tokens

                    if labels[idx, idy] >= self.max_token or self.spread == 0:
                        binned_prob = y_pred[idx, idy, labels[idx, idy]]
                    else:
                        min_label = labels[idx, idy] - self.spread
                        if min_label < 0:
                            min_label = 0

                        max_label = labels[idx, idy] + self.spread
                        if max_label > self.max_token:
                            max_label = self.max_token

                        binned_prob = torch.sum(y_pred[idx, idy, min_label:max_label])

                    # loss[idx, idy] = -1 * torch.log(binned_prob)
                    loss += -1 * torch.log(binned_prob)

        if self.reduction == "mean":
            return loss / torch.sum(labels > 0)
        else:  # sum
            return loss


class FuzzyCrossEntropy(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, labels, vocab_size=None, num_items_in_batch=None):
        pred = torch.softmax(logits, dim=2)

        loss = 0
        mask = labels[:, :, 0, 0] >= 0

        for lvl in [0]:  # range(labels.shape[3]):
            lbl = labels[:, :, :, lvl]

            x = -1 * torch.log(torch.sum(torch.mul(pred, lbl > 0), dim=2))
            loss += torch.nansum(torch.mul(x, mask))

        if self.reduction == "mean":
            return (loss / 4) / torch.sum(mask)
        else:  # sum
            return loss / 4


class BinomialKLLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

        self.pointwise_loss = torch.nn.KLDivLoss(reduction="none")

    def forward(self, logits, labels, vocab_size=None, num_items_in_batch=None):
        pred = torch.log_softmax(logits, dim=2)

        mask = labels <= 1  # use values greater than 1 as a mask token

        pl = torch.mul(self.pointwise_loss(pred, labels), mask)

        if self.reduction == "none":
            return pl
        elif self.reduction == "mean":
            return torch.sum(pl) / torch.sum(mask)
        else:  # sum
            return torch.sum(pl)
