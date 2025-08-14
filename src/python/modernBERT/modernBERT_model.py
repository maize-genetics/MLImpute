import math
from dataclasses import dataclass
from typing import Optional, Literal

import lightning as L
import torch
from lightning.pytorch.utilities import grad_norm
from torch import nn as nn, Tensor
from transformers import ModernBertConfig, ModernBertModel

class SNPLoss(nn.Module):
    def __init__(self):
        super(SNPLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits, unmasked_input):
        targets = (unmasked_input > 0).to(torch.float32)
        return self.loss_fn(logits, targets)

class SNPLossSmoothAll(nn.Module):
    def __init__(self, lambda_smooth=0.2):
        super(SNPLossSmoothAll, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.lambda_smooth = lambda_smooth

    def forward(self, logits, unmasked_input):
        targets = (unmasked_input > 0).to(torch.float32)
        diff = logits[:, 1:] - logits[:, :-1]
        smoothness_penalty = torch.mean(torch.abs(diff))
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        return self.loss_fn(logits, targets) + self.lambda_smooth * smoothness_penalty


@dataclass
class BERTImputeConfig:
    vocab_size: int = 25
    dropout: float = 0.1
    max_sequence_length: Optional[int] = None

    architecture: Literal["encoder-only", "sequence-only", "classifier-only", "all"] = (
        "encoder-only"
    )
    token_embedding_dim: int = 512
    head_encoder_layers: int = 4
    head_encoder_heads: int = 8
    base_encoder_dim: int = 512
    base_encoder_path: str | None = None
    base_encoder_revision: str | None = None
    base_encoder_frozen: bool = True
    hidden_size = 512  # Base encoder hidden size

    train_eval_frequency: Optional[int] = 250
    enable_visualization: bool = True


class BERTImpute(L.LightningModule):
    def __init__(
        self,
        config: BERTImputeConfig,
        learning_rate: float = 8e-4,
        learning_rate_decay: str = "none",
        learning_rate_warmup_ratio: float = 0.1,
        torch_compile: bool = False,
    ):
        super(BERTImpute, self).__init__()
        self.config = config
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_warmup_ratio = learning_rate_warmup_ratio
        self.torch_compile = torch_compile
        self.criterion = SNPLossSmoothAll()
        self.input_proj = nn.Linear(25, self.config.hidden_size)
        self.classifier= nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_size * 2, 25)
        )

        self.head_config = None
        self.head_encoder = None

        self.head_config = ModernBertConfig(
            vocab_size=config.vocab_size,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.hidden_size * 4,
            num_hidden_layers=config.head_encoder_layers,
            num_attention_heads=config.head_encoder_heads,
            pad_token_id=0,
            max_position_embeddings=config.max_sequence_length,
            attention_dropout=self.config.dropout,
            mlp_dropout=self.config.dropout,
        )
        self.head_encoder = ModernBertModel(self.head_config)

        if self.torch_compile:
            # Avoid base model compilation due to https://github.com/pytorch/pytorch/issues/146129
            self.head_encoder = torch.compile(self.head_encoder, fullgraph=False)

        self.save_hyperparameters()

        # -------------------------------------------------------------------------
        # Lightning Methods
        # -------------------------------------------------------------------------

    def on_before_optimizer_step(self, optimizer):
        # See: https://github.com/Lightning-AI/pytorch-lightning/pull/16745
        norms = {
            f"grad/{k}": v
            for k, v in grad_norm(self, norm_type=2).items()
            if "norm_total" in k or "final_norm" in k
        }
        if norms:
            self.log_dict(norms)

    #This is the hook for Lightning so it can do it without all the overhead.
    #Todo copy over in the training logic from lazy_train
    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = self._compute_loss(outputs, batch)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = self._compute_loss(outputs, batch)
        self.log("valid/loss", loss, sync_dist=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )

        expected_steps = self.trainer.estimated_stepping_batches
        warmup_steps = math.ceil(expected_steps * self.learning_rate_warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                if self.learning_rate_decay == "cosine":
                    # Cosine annealing
                    progress = (step - warmup_steps) / (expected_steps - warmup_steps)
                    return 0.5 * (1.0 + math.cos(progress * math.pi))
                elif self.learning_rate_decay == "none":
                    # No decay
                    return 1.0
                else:
                    raise ValueError(
                        f"Invalid learning rate decay: {self.learning_rate_decay}"
                    )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scheduler_config = dict(scheduler=scheduler, interval="step", frequency=1)
        return [optimizer], [scheduler_config]

    # -------------------------------------------------------------------------
    # Model Methods
    # -------------------------------------------------------------------------

    def forward(self, inputs_embeds: Tensor | None = None) -> Tensor:
        B, L, _ = inputs_embeds.shape  # input_features: (B, L, input_dim)
        mask = torch.rand(B, L, device=inputs_embeds.device) < 0.1  # randomly change 10% of input to 0 for training
        input_masked = inputs_embeds.masked_fill(mask.unsqueeze(-1), 0)
        projection = self.input_proj(input_masked)
        hidden = self.head_encoder.forward(
                inputs_embeds=projection
            ).last_hidden_state
        logits = self.classifier(hidden)
        if self.training:
            self.log("train/logit_mean", float(logits.mean()))
        return logits


    def _compute_loss(self, logits, unmasked):
        return self.criterion(logits, unmasked)
