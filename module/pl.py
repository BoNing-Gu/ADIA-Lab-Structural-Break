import os
import copy
import json
import time
from tqdm.auto import tqdm
from itertools import islice

import pandas as pd

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics.functional import auroc

class CollapseAvoidLoss(torch.nn.Module):
    def __init__(self, min_std=0.1, factor=10):
        super().__init__()
        self.min_std = min_std
        self.factor = factor
    
    def forward(self, logits):
        std = torch.std(torch.sigmoid(logits))
        return torch.clamp((self.min_std - std) * self.factor, 0)

class PLModel(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, config):
        super().__init__()
        self.model = model
        self.optimizer_obj = optimizer 
        self.criterion = criterion
        self.collapse_criterion = CollapseAvoidLoss(min_std=0.1, factor=10)
        self.config = config

        self.automatic_optimization = False
        self.validation_step_outputs = []

        self.save_hyperparameters(ignore=[
            "model", "optimizer", "criterion", "validation_step_outputs", "config"
        ])

    def configure_optimizers(self):
        return self.optimizer_obj

    def on_train_start(self):
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
        st = time.time()
        opt = self.optimizers()
        opt.zero_grad()
        logits = self.model(batch['ts'], batch['period'], batch['key_padding_mask'])  # [bs]
        labels = batch['label'].float()  # shape: [bs]
        loss = self.criterion(logits, labels)
        collapse_loss = self.collapse_criterion(logits)
        total_loss = loss + collapse_loss  # 组合损失
        total_loss.backward()
        opt.step()
        et = time.time()

        # Calculate logit statistics
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            logit_mean = logits.mean()
            logit_std = logits.std()
            prob_mean = probs.mean()
            prob_std = probs.std()
            min_prob = probs.min()
            max_prob = probs.max()

        # Log losses 
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_collapse_loss", collapse_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("time_per_step", et - st, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        # Log statistics 
        self.log("stats/logit_mean", logit_mean, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log("stats/logit_std", logit_std, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log("stats/prob_mean", prob_mean, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log("stats/prob_std", prob_std, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log("stats/prob_min", min_prob, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log("stats/prob_max", max_prob, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx != 0:
            return None

        logits = self.model(batch['ts'], batch['period'], batch['key_padding_mask'])  # [bs]
        labels = batch['label'].float()  # [bs]
        probs = torch.sigmoid(logits)    # [bs]

        # 拼接成 [bs, 2]，第0列是概率，第1列是标签
        output = torch.stack([probs.detach(), labels], dim=1)  # [bs, 2]

        return output
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