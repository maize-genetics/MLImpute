from transformers import ModernBertDecoderConfig, VisionEncoderDecoderModel, get_wsd_schedule
from transformers import ViTModel, ModernBertDecoderForCausalLM
import torch
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np
import torch.optim as optim

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

image_size = 384
num_parents = 24

# for processing the "images"
chunks = image_size // num_parents
pos_length = image_size * chunks

config_decoder = ModernBertDecoderConfig(vocab_size=pos_length + 3, pad_token_id=pos_length, eos_token_id=pos_length + 1,
                                  bos_token_id=pos_length + 2, cls_token_id=pos_length + 1, sep_token_id=pos_length + 2)

encoder = ViTModel.from_pretrained("google/vit-base-patch16-384")
decoder = ModernBertDecoderForCausalLM(config_decoder)

model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
model.config.decoder_start_token_id = pos_length+2
model.config.pad_token_id = pos_length
model.config.vocab_size = pos_length+3
model.config.eos_token_id = pos_length+1


class SegmentationDataset(Dataset):
    def __init__(self, input_file, label_file, image_size=384, num_parents=24, step_size=1536):
        self.window_size = (image_size * image_size) // num_parents
        self.image_size = image_size
        self.num_parents = num_parents
        self.step_size = step_size
        self.matrix = np.load(input_file)[:, 0:num_parents]
        # self.labels = np.load(label_file)

        self.n_windows = (self.matrix.shape[0] - self.window_size) // step_size

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        pos_start = idx * self.step_size
        pos_end = pos_start + self.window_size

        window = np.hstack(np.split(self.matrix[pos_start:pos_end], self.window_size // self.image_size))

        # TODO: normalize
        # ViT wants rgb channels each with mean 0.5 and sd 0.5

        # TODO: import real junction data

        junctions = np.random.randint(0, self.window_size, np.random.randint(0, 10))

        if junctions.shape[0] < 14:
            junctions = np.concatenate(
                ([self.window_size + 2], junctions, [self.window_size + 1], [3456] * (14 - junctions.shape[0])))
        else:
            junctions = np.concatenate(([self.window_size + 2], junctions, [self.window_size + 1]))

        return {
            'pixel_values': torch.tensor(window.unsqueeze(0), dtype=torch.float),
            'labels': torch.tensor(junctions).squeeze()
        }


dataset = SegmentationDataset("/workdir/ahb232/MLImpute/src/training_data/train/CML442_matrix.npy", "bar.txt")
dataset_chunks = torch.utils.data.random_split(dataset, [0.9, 0.1])
dataset_train = dataset_chunks[0]
dataset_val = dataset_chunks[1]

optimizer = optim.AdamW(model.parameters())
lr_scheduler = get_wsd_schedule(optimizer, 100, 100, num_training_steps=len(dataset_train) // 16)

model = model.to(device)
model.train()
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    #    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    optimizers=(optimizer, lr_scheduler)
)
trainer.train()
