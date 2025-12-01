import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers.data.data_collator import InputDataClass
from typing import Any
from transformers.data.data_collator import default_data_collator
from transformers import BertConfig, get_wsd_schedule, BertLMHeadModel, GenerationMixin
import argparse
import sys
from transformers import Trainer, TrainingArguments
import torch.optim as optim
from transformers import PreTrainedModel


class DecoderOnlyConfig(BertConfig):
    def __init__(self, num_parents=24, **kwargs):
        self.num_parents = num_parents
        super().__init__(**kwargs)


class DecoderOnlyModel(PreTrainedModel, GenerationMixin):
    config_class = DecoderOnlyConfig

    def __init__(self, config):
        super().__init__(config)
        self.proj = torch.nn.Linear(config.num_parents, config.hidden_size)
        self.model = BertLMHeadModel(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cache_position=None,
            **loss_kwargs,
    ):
        if encoder_hidden_states is not None:
            projection = self.proj(encoder_hidden_states)
        else:
            projection = encoder_hidden_states

        return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                          position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                          encoder_hidden_states=projection, encoder_attention_mask=encoder_attention_mask,
                          labels=labels, past_key_values=past_key_values, use_cache=use_cache,
                          output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                          return_dict=return_dict, cache_position=cache_position, **loss_kwargs)


class DecoderOnlyDataset(Dataset):
    def __init__(self, keyfile, window_size=256, num_parents=24, step_size=128, windows=None, split_norm_levels=False,
                 include_index=False):
        self.keyfile = pd.read_csv(keyfile, sep="\t")
        self.window_size = window_size
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
        window = (matrix - mean) / sd

        # We've got labels and decoder input id's separately here, even though they should just be
        # shifted versions of one another. This was part of debugging that I did because
        # I think the default shifting function built into the model isn't working right
        # for our needs. Specifying both overrides the default though.
        input_ids = np.concatenate(([self.window_size + 2], junctions))
        labels0 = np.concatenate((junctions, [self.window_size + 1]))

        mask = np.ones(len(junctions) + 1)

        # Note: we are relying on the data collator to handle padding, so labels do not have
        # a fixed length

        return {
            'encoder_hidden_states': torch.tensor(window, dtype=torch.float),  # (3, image_size, image_size)
            'labels': torch.tensor(labels0),
            'attention_mask': mask,  # (boolean, same length as labels)
            'input_ids': torch.tensor(input_ids),  # (torch.int64, same length as labels)
            'file_idx': file_idx,
            'pos_idx': pos_idx
        }


def decoder_data_collator(features: list[InputDataClass]) -> dict[str, Any]:
    # loss calculations will ignore this token
    padding_token = -100
    input_padding_token = 256  # TODO remove hard-coding

    # pad label features to the length of the longest
    # right-padded for training
    longest_seq = np.max([feat["labels"].shape[0] for feat in features])

    for feat in features:
        if feat["labels"].shape[0] < longest_seq:
            pad_len = longest_seq - feat["labels"].shape[0]
            feat["labels"] = np.concatenate((feat["labels"], [padding_token] * pad_len))
            feat["input_ids"] = np.concatenate((feat["input_ids"], [input_padding_token] * pad_len))
            feat["attention_mask"] = np.concatenate((feat["attention_mask"], np.zeros(pad_len)))

        feat["labels"] = torch.tensor(feat["labels"], dtype=torch.int64).clone()
        feat["input_ids"] = torch.tensor(feat["input_ids"], dtype=torch.int64).clone()

    # pass padded tokens to default collator
    batch = default_data_collator(features)

    return batch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-parents", type=int, default=24, help="number of parents")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to a previous training checkpoint")
    parser.add_argument("--keyfile", type=str, required=True, help="keyfile with list of input files")
    parser.add_argument("--windows", type=str, default=None, help="Optional, specify which windows to include")
    parser.add_argument("--num-epochs", "-e", type=int, default=2, help="number of training epochs")
    parser.add_argument("--skip-warmup", action="store_true", help="skip the warmup stage of WSD")
    parser.add_argument("--allow-cpu", action="store_true", help="allow cpu for training (not recommended)")
    parser.add_argument("--run-name", "--rn", type=str, default="run-1", help="wandb run name")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="batch size")
    parser.add_argument("--save-model-path", "-s", type=str, default="output_model",
                        help="path to save the best performing model")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="max learning rate")
    # TODO: allow more control over fuzzy loss
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        device = torch.device('cuda')
    else:
        if args.allow_cpu:
            device = torch.device('cpu')
            num_devices = 1
        else:
            print("Error: GPU not found")
            sys.exit()

    # for processing the "images"
    pos_length = 256

    # WSD is suited for picking up from a previous training checkpoint
    # So, we make this an option
    if args.checkpoint is not None:
        model = BertLMHeadModel.from_pretrained(args.checkpoint)
    else:
        # If no checkpoint is provided, we initialize a new model
        config = DecoderOnlyConfig(num_parents=args.num_parents, vocab_size=pos_length + 3, max_position_embeddings=256,
                                   is_decoder=True, pad_token_id=pos_length, eos_token_id=pos_length + 1,
                                   bos_token_id=pos_length + 2, cls_token_id=pos_length + 1, sep_token_id=pos_length + 2,
                                   add_cross_attention=True, num_hidden_layers=6, num_attention_heads=12)

        model = DecoderOnlyModel(config)

    dataset = DecoderOnlyDataset(args.keyfile, windows=args.windows)

    dataset_chunks = torch.utils.data.random_split(dataset, [0.95, 0.05])
    dataset_train = dataset_chunks[0]
    dataset_val = dataset_chunks[1]

    # set up model
    model = model.to(device)
    model.train()

    # set up optimizer and scheduler
    # we pre-calculate the number of training and warmup steps needed
    # based on batch size
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 10% warmup/decay
    len_warmup = (len(dataset_train) * args.num_epochs) // (args.batch_size * 10 * num_devices)
    if args.skip_warmup:
        lr_scheduler = get_wsd_schedule(optimizer, 0, len_warmup,
                                        num_training_steps=(args.num_epochs * len(dataset_train)) // (
                                                args.batch_size * num_devices))
    else:
        lr_scheduler = get_wsd_schedule(optimizer, len_warmup, len_warmup,
                                        num_training_steps=(args.num_epochs * len(dataset_train)) // (
                                                args.batch_size * num_devices))

    # use huggingface trainer to avoid boilerplate code
    training_args = TrainingArguments(
        output_dir=args.save_model_path,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=2000,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        report_to="wandb",
        run_name=args.run_name
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=decoder_data_collator,
        optimizers=(optimizer, lr_scheduler)
    )
    trainer.train()


if __name__ == '__main__':
    main()
