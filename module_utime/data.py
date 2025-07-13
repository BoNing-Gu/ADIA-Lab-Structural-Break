import pandas as pd
import numpy as np
from typing import Dict, Any
from pathlib import Path
from tqdm.auto import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from datasets import Dataset
    
# -------------------------
# Dataset & Model Preparation
# -------------------------
def process_func(X_df, y_value=None, config=None):
    X_df = X_df.sort_index(level="time")
    id_ = X_df.index.get_level_values("id")[0]
    seq_len = len(X_df)

    if y_value is not None:
        # y_value为True->period就作为label / y_value为False->全部为0
        label = X_df['period'].values.astype(np.float32) if y_value else np.zeros(seq_len, dtype=np.float32)
    else:
        label = None

    result = {
        'id': id_,
        'ts': X_df['value'].values.astype(np.float32),           # shape: [max_length]
        'label': label
    }

    return result

def create_structured_dataset(X_df, y_series, config=None):
    grouped = X_df.groupby('id')
    processed = [
        process_func(group, y_series.loc[id_], config=config)
        for id_, group in tqdm(grouped, desc="Processing dataset")
    ]
    return Dataset.from_list(processed)

def collate_fn(batch):
    ts = [torch.tensor(item['ts'], dtype=torch.float) for item in batch]
    label = [torch.tensor(item['label'], dtype=torch.float) for item in batch] if 'label' in batch[0] else None
    id_ = torch.tensor([item['id'] for item in batch], dtype=torch.long)

    padded_ts = pad_sequence(ts, batch_first=True)  # [B, L] or [B, L, D]
    padded_label = pad_sequence(label, batch_first=True) if label is not None else None

    return {
        'ts': padded_ts.unsqueeze(-1),  # [B, L, nvars=1]
        'label': padded_label,          # [B, L]
        'id': id_
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
            collate_fn=collate_fn
        )

    def valid_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,  
            collate_fn=collate_fn
        )