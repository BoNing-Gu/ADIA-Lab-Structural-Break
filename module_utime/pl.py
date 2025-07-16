import os
import copy
import json
import time
from tqdm.auto import tqdm
from itertools import islice

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics.functional import auroc
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from lightning.pytorch.utilities import grad_norm

class CollapseAvoidLoss(torch.nn.Module):
    def __init__(self, min_std=0.1, factor=10):
        super().__init__()
        self.min_std = min_std
        self.factor = factor
    
    def forward(self, logits):
        std = torch.std(torch.sigmoid(logits))
        return torch.clamp((self.min_std - std) * self.factor, 0)

# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, 2, L)` where N is the batch size and L is the sequence length
        - Target: :math:`(N, L)` where each value is :math:`0 ≤ targets[i] ≤ 1`.
    """

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError(
                "Input type is not a torch.Tensor. Got {}".format(type(input))
            )

        if not len(input.shape) == 3:
            raise ValueError(
                "Invalid input shape, we expect (N, 2, L). Got: {}".format(input.shape)
            )

        if not input.shape[-1] == target.shape[-1]:
            raise ValueError(
                "input and target shapes must be the same. Got: {}".format(
                    input.shape, input.shape
                )
            )

        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}".format(
                    input.device, target.device
                )
            )

        # get positive prediction only
        positive_preds = input[:, 1, :]

        # compute the actual dice score
        intersection = torch.sum(positive_preds * target, (1))
        union = torch.sum(positive_preds + target, (1))

        dice_score = 2.0 * intersection / (union + self.eps)

        return torch.mean(1.0 - dice_score)

class PLModel(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.criterion = DiceLoss()
        # self.collapse_criterion = CollapseAvoidLoss(min_std=0.1, factor=10)
        self.config = config

        self.automatic_optimization = True
        self.validation_step_outputs = []

        self.save_hyperparameters(ignore=[
            "model", "validation_step_outputs", "config"
        ])

    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self, norm_type=2)

    #     self.log_dict(norms)

    def configure_optimizers(self):
        if self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)
        res = {"optimizer": self.optimizer}

        if self.config.lr_scheduler == "CLR":
            lr_scheduler = CyclicLR(
                optimizer=self.optimizer,
                base_lr=self.config.learning_rate / 10.0,
                max_lr=self.config.learning_rate,
                step_size_up=getattr(self.config, "clr_step_size_up", 5),
                mode="triangular",
                cycle_momentum=self.config.optimizer != "Adam",
                # verbose=False,
            )
            res["lr_scheduler"] = {
                "name": "CyclicLR",
                "scheduler": lr_scheduler,
                "interval": "step",
            }
        elif self.config.lr_scheduler == "ROP":
            lr_scheduler = ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode="min",
                factor=self.config.get("rop_factor", 0.5),
                patience=25,
                min_lr=1e-6,
                # verbose=False,
            )
            res["lr_scheduler"] = {
                "name": "ReduceLROnPlateau",
                "scheduler": lr_scheduler,
                "strict": True,
                "monitor": "validation/loss_smooth",
            }

        return res

    def save_plot_prediction(self, name, X, y, preds, metadatas):
        """
        Plot a random event and upload it to neptune for debugging of the training
        """
        # Select random batch item
        _idx = np.random.randint(0, X.shape[0])
        event_id, (start_index) = metadatas[_idx]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f"{event_id=}, {start_index=}", fontsize=16)
        ax.plot(X[_idx, ::5, 0].cpu(), color="black")
        ax.plot(y[_idx, ::5].cpu() * 50, color="green")
        ax.plot(preds[_idx, 1, ::5].cpu() * 50, color="red")
        fig.tight_layout()
        plt.close(fig)

        self.logger.experiment[name].append(fig)

    def training_step(self, batch, batch_idx):
        inputs = batch['ts']
        targets = batch['target']
        metadata = batch['id'] 

        preds = self.model(inputs)

        loss = self.criterion(preds, targets)

        self.log("training/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return {"loss": loss, "predicted": preds, "targets": targets}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx != 0:
            return None

        inputs = batch['ts']         # [B, L, 2]
        targets = batch['target']    # [B, L]
        metadata = batch['id']
        labels = batch['label']

        preds = self.model(inputs)   # [B, 2, L]
        loss = self.criterion(preds, targets)

        self.log("validation/loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)

        probs = preds[:, 1, :]       # [B, L]
        periods = inputs[:, :, 1]    # [B, L]

        results = []

        for i in range(inputs.size(0)):
            period_series = periods[i]
            pred_series = probs[i]

            transitions = (period_series[:-1] == 0) & (period_series[1:] == 1)
            transition_idx = torch.where(transitions)[0] + 1

            if len(transition_idx) == 0:
                continue

            t = transition_idx[0]

            pred_prob = pred_series[t].item()
            real_label = labels[i].item()
            results.append([pred_prob, real_label])

        if len(results) == 0:
            return torch.empty((0, 2), device=self.device)

        return torch.tensor(results, dtype=torch.float32, device=self.device)  # shape: [n, 2]

    def on_validation_batch_end(self, validation_step_outputs, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx != 0:
            return None
        validation_step_outputs = self.trainer.strategy.all_gather(validation_step_outputs)
        # print(f"[Debug] Rank {self.trainer.global_rank} collected {len(validation_step_outputs)} batches")
        if self.trainer.is_global_zero:
            # print(f"\n[Debug] Collected {len(validation_step_outputs)} test batches (from all GPUs)")
            # for i, batch_output in enumerate(validation_step_outputs):
            #     print(f"\n[Debug] --- Batch {i} ---")
                
            #     print(f"[Debug] batch output shape: {batch_output.shape} | dtype: {batch_output.dtype}")
            self.validation_step_outputs.append(validation_step_outputs)
        return validation_step_outputs
    
    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            validation_step_outputs = [
            batch.reshape(-1, batch.shape[-1])
                for batch in self.validation_step_outputs
            ]
            all_outputs = torch.cat(validation_step_outputs, dim=0)  # [total_samples, 2]

            all_probs = all_outputs[:, 0]    # 概率列
            all_labels = all_outputs[:, 1].long()  # 标签列，转成整型
            
            try:
                auc = auroc(all_probs, all_labels.int(), task="binary")
            except Exception as e:
                auc = torch.tensor(0.0).to(self.device)
                print(f"Warning: AUC computation failed: {e}")

            self.log("val_auc", auc, prog_bar=True, sync_dist=True)

            self.validation_step_outputs.clear()
