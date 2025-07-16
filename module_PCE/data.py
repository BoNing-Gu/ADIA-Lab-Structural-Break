import pandas as pd
import numpy as np
from typing import Dict, Any
from pathlib import Path
from tqdm.auto import tqdm

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from datasets import Dataset
    
# -------------------------
# Dataset & Model Preparation
# -------------------------
def process_func(X_df, y_value=None, config=None):
    max_length = config.seq_length
    X_df = X_df.sort_index(level="time")
    X_df['value'] = X_df['value'].cumsum()
    id_ = X_df.index.get_level_values("id")[0]
    seq_len = len(X_df)
    X_df['key_padding_mask'] = 0

    if seq_len > max_length:
        # truncate
        X_df = X_df.iloc[-max_length:]
        seq_len = max_length
    if seq_len < max_length:
        # padding
        padding_len = max_length - seq_len
        last_time = X_df.index.get_level_values('time')[-1]
        padding_times = [last_time + i for i in range(1, padding_len + 1)]

        padding_df = pd.DataFrame({
            'value': [0.0] * padding_len,
            'period': [1] * padding_len, 
            'key_padding_mask': [1] * padding_len
        }, index=pd.MultiIndex.from_arrays(
            [[id_] * padding_len, padding_times],
            names=["id", "time"]
        ))

        X_df = pd.concat([X_df, padding_df])

    result = {
        'id': id_,
        'ts': X_df['value'].values.astype(np.float32),           # shape: [max_length]
        'period': X_df['period'].values.astype(np.float32),
        'key_padding_mask': X_df['key_padding_mask'].values.astype(bool)
    }
    if y_value is not None:
        result['label'] = y_value
        # result['label'] = encode_label(config.num_classes, int(y_value))

    return result

def encode_label(num_classes, label):
    """Encode label to one-hot vector"""
    label_to_idx = {label: idx for idx, label in enumerate(num_classes)}
    target = np.zeros(num_classes)
    if label in label_to_idx:
        target[label_to_idx[label]] = 1.0
    return target

def create_structured_dataset(X_df, y_series, config=None):
    grouped = X_df.groupby('id')
    processed = [
        process_func(group, y_series.loc[id_], config=config)
        for id_, group in tqdm(grouped, desc="Processing dataset")
    ]
    return Dataset.from_list(processed)

def collate_fn(batch):
    ts = torch.tensor([item['ts'] for item in batch], dtype=torch.float)
    period = torch.tensor([item['period'] for item in batch], dtype=torch.long)
    key_padding_mask = torch.tensor([item['key_padding_mask'] for item in batch], dtype=torch.bool)
    label = torch.tensor([item['label'] for item in batch], dtype=torch.long) if 'label' in batch[0] else None

    return {
        'ts': ts.unsqueeze(-1),                # [B, L, 1]
        'period': period,                      # [B, L]
        'key_padding_mask': key_padding_mask,  # [B, L]
        'label': label,                        # [B]
    }
    
# -------------------------
# Lightning DataModule
# -------------------------
class LightDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, valid_dataset, train_batch_size=4, valid_batch_size=8, num_workers=4):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True, 
            collate_fn=lambda batch: collate_fn(batch)
        )

    def valid_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,  
            collate_fn=lambda batch: collate_fn(batch)
        )