from transformers import ModernBertDecoderConfig, VisionEncoderDecoderModel, get_wsd_schedule
from transformers import ViTModel, ModernBertDecoderForCausalLM
import torch
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np
import torch.optim as optim
from transformers.data.data_collator import InputDataClass
from typing import Any
from transformers.data.data_collator import default_data_collator

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

image_size = 384
num_parents = 24

# for processing the "images"
chunks = image_size // num_parents
pos_length = image_size * chunks

pretrained_path=None

if pretrained_path is not None:
    model = VisionEncoderDecoderModel.from_pretrained(pretrained_path)
else:
    config_decoder = ModernBertDecoderConfig(vocab_size=pos_length + 3, pad_token_id=pos_length, eos_token_id=pos_length + 1,
                                      bos_token_id=pos_length + 2, cls_token_id=pos_length + 1, sep_token_id=pos_length + 2)

    encoder = ViTModel.from_pretrained("google/vit-base-patch16-384")
    decoder = ModernBertDecoderForCausalLM(config_decoder)

    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.config.decoder_start_token_id = pos_length+2
    model.config.pad_token_id = pos_length
    model.config.vocab_size = pos_length+3
    model.config.eos_token_id = pos_length+1


def custom_data_collator(features: list[InputDataClass]) -> dict[str, Any]:
    batch = default_data_collator(features)
    batch["return_dict"] = False

    return batch

class SegmentationDataset(Dataset):
    def __init__(self, input_file_names, image_size=384, num_parents=24, step_size=1536):
        self.input_file_names = input_file_names
        self.window_size = (image_size * image_size) // num_parents
        self.image_size = image_size
        self.num_parents = num_parents
        self.step_size = step_size
        self.windows = self.__generate_windows__()

        self.n_windows = len(self.windows)

    def __generate_windows__(self):
        windows = [] # file idx, window step idx

        for idx in range(len(self.input_file_names)):
            filelen = np.load(self.input_file_names[idx]).shape[0]

            num_windows = (filelen - self.window_size) // self.step_size

            windows.extend([(idx, idy) for idy in range(num_windows)])

        return windows

    def __len__(self):
        return self.n_windows

    def __bins_to_idx__(self, labels_binned):
        return [idx+1 for idx in range(labels_binned.shape[0] - 1) if labels_binned[idx] != labels_binned[idx+1]]

    def __getitem__(self, idx):
        file_idx, pos_idx = self.windows[idx]

        pos_start = pos_idx * self.step_size
        pos_end = pos_start + self.window_size

        ip = np.load(self.input_file_names[file_idx], allow_pickle=True, mmap_mode='r')[pos_start:pos_end]

        matrix = ip[:, 0:num_parents]

        # normalize the matrix according to ViT's requirements (mean 0.5 std 0.5)
        mean = np.mean(matrix)
        sd = np.std(matrix)

        matrix = (matrix - mean) / sd

        matrix = matrix * 0.5 + 0.5

        labels_binned = ip[:, num_parents]

        junctions = self.__bins_to_idx__(labels_binned)

        window = np.hstack(np.split(matrix, self.window_size // self.image_size))

        # for now, create greyscale image by stacking the matrix 3 times
        # later we'll try encoding different information in each layer

        window = np.stack((window, window, window), axis=0)

        if len(junctions) < 62:
            mask = np.concatenate((np.zeros(62 - len(junctions)), np.ones(len(junctions) + 2)))
            junctions = np.concatenate(
                ([self.window_size] * (62 - len(junctions)), [self.window_size + 2], junctions, [self.window_size + 1]))
        else:
            mask = np.ones(64)
            junctions = np.concatenate(([self.window_size + 2], junctions, [self.window_size + 1]))

        return {
            'pixel_values': torch.tensor(window, dtype=torch.float),
            'labels': torch.tensor(junctions, dtype=torch.int64).squeeze(),
            'decoder_attention_mask': mask
        }


dataset = SegmentationDataset(["/workdir/ahb232/MLImpute/data/mock_data/sample_A.npy",
                               "/workdir/ahb232/MLImpute/data/mock_data/sample_B.npy",
                               "/workdir/ahb232/MLImpute/data/mock_data/sample_C.npy"])

dataset_chunks = torch.utils.data.random_split(dataset, [0.9, 0.1])
dataset_train = dataset_chunks[0]
dataset_val = dataset_chunks[1]

optimizer = optim.AdamW(model.parameters())

model = model.to(device)
model.train()

num_epochs = 2

len_warmup = (len(dataset_train) * num_epochs) // 320

lr_scheduler = get_wsd_schedule(optimizer, len_warmup, len_warmup, num_training_steps=(num_epochs * len(dataset_train)) // 32)

training_args = TrainingArguments(
    output_dir='./results_epoch_0_1',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    report_to="wandb",
    run_name="test_epochs_0-1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    data_collator=custom_data_collator,
    optimizers=(optimizer, lr_scheduler)
)
trainer.train()
