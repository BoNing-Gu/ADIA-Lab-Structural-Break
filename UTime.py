#!/usr/bin/env python
# coding: utf-8
import os
import glob
import typing
from typing import Callable, Optional

# Import your dependencies
import math
import joblib
import pandas as pd
import numpy as np
import scipy
import sklearn.metrics

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import lightning.pytorch as pl

from datasets import Dataset
from module_utime.data import create_structured_dataset, LightDataModule
from module_utime.UTime import Model
from module_utime.pl import PLModel
from module_utime.logger import CustomLogger
from swanlab.integration.pytorch_lightning import SwanLabLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint

# -------------------------
# Main Training
# -------------------------
def main(config):
    # Load & Preprocess Dataset
    X_train, y_train = pd.read_parquet(os.path.join(config.data_path, 'X_train.parquet')), pd.read_parquet(os.path.join(config.data_path, 'y_train.parquet'))['structural_breakpoint']
    # X_test, y_test = pd.read_parquet(os.path.join(config.data_path, 'X_test.reduced.parquet')), pd.read_parquet(os.path.join(config.data_path, 'y_test.reduced.parquet'))['structural_breakpoint']
    train_dataset = create_structured_dataset(X_train, y_train, config=config)
    split_dataset = train_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset['train']
    valid_dataset = split_dataset['test']
    print(train_dataset)
    print(valid_dataset)

    # Create DataModule
    dm = LightDataModule(train_dataset, valid_dataset, config.train_batch_size, config.valid_batch_size, config.num_workers)
    train_dataloader = dm.train_dataloader()
    valid_dataloader = dm.valid_dataloader()
    
    # Config Model
    model = Model(config)
    model.train()   
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params} / 总参数: {total_params} (占比: {100*trainable_params/total_params:.2f}%)")
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print(f"❌ 不可训练参数: {name}, shape={param.shape}, total={param.numel()}")
    
    # Initialize PLModel
    pl_model = PLModel(
        model=model, 
        config=config
    )

    # csv_logger = CustomLogger(
    #     save_dir=os.path.join(config.log_dir, config.model_name), 
    #     version=config.version,
    #     resume=config.resume
    # )
    swanlab_logger = SwanLabLogger(
        project=f"{config.model_name}",
        experiment_name=f"{config.version}"
    )
    # checkpoint_callback = ModelCheckpoint(
    #     every_n_epochs=1,
    #     monitor="validation/loss",
    #     save_top_k=3,
    #     dirpath=os.path.join(config.ckpt_dir, config.model_name, config.version),
    #     filename='epoch{epoch:02d}-valauc{val_auc:.4f}'
    # )
    # if config.resume:
    #     ckpt_dir = checkpoint_callback.dirpath
    #     list_of_files = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))  
    #     if list_of_files:
    #         ckpt_path = max(list_of_files, key=os.path.getctime)
    #         print(f"Resuming from checkpoint: {ckpt_path}")
    #     else:
    #         ckpt_path = None
    # else:
    #     ckpt_path = None
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config.device,
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=config.epochs,
        precision="32",
        logger=[swanlab_logger],  # csv_logger
        log_every_n_steps=1,
        # enable_checkpointing=True,
        # callbacks=[checkpoint_callback],
        # accumulate_grad_batches=config.grad_accumulate  # dp cannot use grad accumulate
    )
    
    # Train
    trainer.fit(
        model=pl_model, 
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        # ckpt_path=ckpt_path
    )

class Config:
    def __init__(self):
        # Basic
        self.seed = 42
        self.device = "cuda"  # or "cpu"
        self.num_workers = 4
        
        # Data
        self.data_path = "./data"
        self.n_vars = 2   
        self.train_batch_size = 128
        self.valid_batch_size = 128
        self.spike_std = 64
        
        # Model
        self.model_name = "UTime"
        self.num_classes = 2
        ## structure
        self.in_channels = 2
        self.network_depth = 4
        self.channels_size = 128
        self.filters = [
            self.channels_size * 2**i for i in range(self.network_depth)
        ]
        self.maxpool_kernels = [6, 6, 4, 2]
        self.kernel_size = 4
        self.dilation = 2
        self.dropout = 0.1

        # Training
        self.device = [0]
        self.verbose = False
        self.epochs = 200
        self.optimizer = "Adam" # or SGD
        self.lr_scheduler = "CLR"  # or "ROP"
        self.learning_rate = 1e-4
        self.momentum = 0.9
        self.weight_decay = 1e-5
        self.val_loss_smooth_param = 0.25
        self.ckpt_dir = "./checkpoints"
        self.log_dir = "./logs"
        self.version = "exp2"
        self.resume = False
        

if __name__ == "__main__":
    config = Config()
    pl.seed_everything(config.seed)

    main(config)