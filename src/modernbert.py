import os
import time
import lightning as L
import torch
import torch.nn as nn
import numpy as np
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
import warnings
from torch import set_float32_matmul_precision
import logging
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader
from bimamba_train import WindowIndexDataset, gather_npy_paths

from python.modernBERT.modernBERT_model import BERTImputeConfig, BERTImpute

with open("wandb_key.txt", 'r') as f:
    key = f.read().strip()

wandb.login(key=key)

class RankZeroLogger:
    def __init__(self, logger):
        self._logger = logger

    def debug(self, *args, **kwargs):
        return rank_zero_only(self._logger.debug)(*args, **kwargs)

    def info(self, *args, **kwargs):
        return rank_zero_only(self._logger.info)(*args, **kwargs)

    def warning(self, *args, **kwargs):
        return rank_zero_only(self._logger.warning)(*args, **kwargs)

    def error(self, *args, **kwargs):
        return rank_zero_only(self._logger.error)(*args, **kwargs)

def rank_zero_logger(logger: logging.Logger) -> RankZeroLogger:
    return RankZeroLogger(logger)

logger = rank_zero_logger(logging.getLogger(__name__))

class SNPLoss(nn.Module):
    def __init__(self):
        super(SNPLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits, unmasked_input):
        targets = (unmasked_input > 0).to(torch.float32)
        return self.loss_fn(logits, targets)


def train_model():
    checkpoint = "saved_models/"
    seed = 12345
    checkpoint_frequency = 5000
    strategy = "ddp_find_unused_parameters_true"

    epochs = 10
    output_dir="outputfinal/"
    log_frequency = 5
    gpu = 1
    num_nodes = 1
    val_check_interval = .25
    limit_val_batches = 1.0
    accumulate_grad_batches = 64
    batch_size = 64

    window_size = 512
    train_eval_frequency = 250
    head_encoder_layers = 2

    learning_rate = 8e-4
    learning_rate_decay = "none"
    torch_compile = "no"

    # Suppress Lightning warnings about sync_dist (these metrics are fine as averages across batches); e.g.:
    warnings.filterwarnings(
        "ignore", ".*It is recommended to use `self.log\(.*sync_dist=True\).*"
    )

    # Set precision and random seed
    set_float32_matmul_precision("medium")
    L.seed_everything(seed)

    train_paths = gather_npy_paths("training_data/train")
    test_paths = gather_npy_paths("training_data/test")

    train_dataset = WindowIndexDataset(train_paths, window_size=window_size, top_n=25,
                                   step_size=window_size, return_decode=False)
    test_dataset = WindowIndexDataset(test_paths, window_size=window_size, top_n=25,
                                  step_size=window_size, return_decode=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    # Reconstruct test_matrix just for SNP accuracy computation
    test_matrix_parts = []
    for path in test_paths:
        matrix = np.load(path, allow_pickle=True, mmap_mode='r')
        end = matrix.shape[0] - (matrix.shape[0] % window_size)
        truncated_matrix = matrix[:end]
        test_matrix_parts.append(truncated_matrix)

    test_matrix = np.concatenate(test_matrix_parts, axis=0)
    test_matrix = torch.tensor(test_matrix, dtype=torch.float32)


    # setup wandb logger
    logger.info(
        f"Setting up WandB logger with project=PHG_Imputation, run=SimpleModernBERT_sarah"
    )

    wandb_logger = WandbLogger(
        name="SimpleModernBERT_sarah",
        project="PHG_Imputation",
        entity="maize-genetics",
        save_dir=output_dir,
    )
    csv_dir = os.path.join(output_dir, "logs/csv")
    logger.info(f"Setting up CSV logger with save_dir={csv_dir}")
    csv_logger = CSVLogger(save_dir=csv_dir)
    loggers = [wandb_logger, csv_logger]

    #initialize model
    logger.info(
        f"Creating new model (architecture=encoder_only, head_encoder_layers={head_encoder_layers})"
    )

            # attention_dropout=self.config.dropout,
            # mlp_dropout=self.config.dropout,
    config = BERTImputeConfig(
        architecture="encoder-only",
        max_sequence_length=window_size,

    )
    model = BERTImpute(
        config,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        torch_compile=torch_compile == "yes",
    )
        # Setup callbacks
    logger.info("Setting up callbacks")
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=checkpoint_frequency,
        save_on_train_epoch_end=True,
        save_last=True,
        save_top_k=-1,
        mode="min",
        monitor="valid/loss",
        auto_insert_metric_name=False,
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    callbacks = [checkpoint_callback, lr_monitor_callback]


    # Initialize trainer
    logger.info("Initializing Lightning Trainer")
    trainer = L.Trainer(
        max_epochs=epochs,
        default_root_dir=output_dir,
        log_every_n_steps=log_frequency,
        precision="bf16-mixed",
        devices=gpu,
        num_nodes=num_nodes,
        gradient_clip_val=1.0,
        strategy=strategy,
        # accelerator="gpu",
        accelerator="gpu",
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
        accumulate_grad_batches=accumulate_grad_batches,
        logger=loggers,
        callbacks=callbacks,
        profiler="simple",
        deterministic=False,
    )

    # Start training
    logger.info("Starting training...")
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
        #ckpt_path=checkpoint,
    )

    logger.info(f"Training complete (see {output_dir})")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    start_time = time.time()
    train_model()
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
