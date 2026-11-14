import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data_module import EPSepsisDataset, collate_ep_batch
from .model import FocalLoss, SepsisLSTM, SepsisTransformer


class EPSepsisLightningModule(L.LightningModule):
    def __init__(self, architecture_name, input_dim, lr=1e-3, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.architecture_name = architecture_name
        self.lr = lr

        if architecture_name.startswith("lstm"):
            self.model = SepsisLSTM(
                input_dim=input_dim,
                hidden_dim=kwargs.get("hidden_dim", 64),
                num_layers=kwargs.get("num_layers", 2),
                dropout=kwargs.get("dropout", 0.2),
                use_dual_pooling=kwargs.get("use_dual_pooling", True),
                use_tcn_conv=kwargs.get("use_tcn_conv", False),
                bidirectional=kwargs.get("bidirectional", False),
            )
        elif architecture_name.startswith("transformer"):
            self.model = SepsisTransformer(
                input_dim=input_dim,
                d_model=kwargs.get("d_model", 64),
                nhead=kwargs.get("nhead", 4),
                num_layers=kwargs.get("num_layers", 2),
                dim_feedforward=kwargs.get("dim_feedforward", 128),
                dropout=kwargs.get("dropout", 0.1),
                use_dual_pooling=kwargs.get("use_dual_pooling", True),
                pos_type=kwargs.get("pos_type", "learned"),
                max_len=240,
                use_cls_token=kwargs.get("use_cls_token", True),
                use_tcn_conv=kwargs.get("use_tcn_conv", False),
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture_name}")

        use_focal_loss = kwargs.get("use_focal_loss", False)
        pos_weight = kwargs.get("pos_weight", 1.0)
        if isinstance(pos_weight, float):
            pos_weight = torch.tensor([pos_weight], dtype=torch.float32)

        if use_focal_loss:
            self.loss_fn = FocalLoss(pos_weight=pos_weight, gamma=2.0)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, x, lengths=None, padding_mask=None):
        if self.architecture_name.startswith("lstm"):
            return self.model(x, lengths)
        else:
            return self.model(x, padding_mask)

    def training_step(self, batch, batch_idx):
        x, y, lengths, padding_mask = batch
        logits = self(x, lengths=lengths, padding_mask=padding_mask).squeeze(-1)
        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, lengths, padding_mask = batch
        logits = self(x, lengths=lengths, padding_mask=padding_mask).squeeze(-1)
        loss = self.loss_fn(logits, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        weight_decay = self.hparams.get("weight_decay", 1e-4)
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=weight_decay)


def build_ep_trainer(epochs, device_str):
    import sys

    trainer_kwargs = {
        "max_epochs": epochs,
        "accelerator": "auto",
        "devices": 1,
        "enable_progress_bar": False,
        "logger": False,
        "enable_checkpointing": False,
    }
    if "cuda" in device_str:
        torch.backends.cudnn.benchmark = True
        trainer_kwargs["precision"] = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    return L.Trainer(**trainer_kwargs)
