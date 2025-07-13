import pandas as pd
import numpy as np
import scipy.stats
from typing import Dict, Any
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
from torch.nn.utils.rnn import pad_sequence
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from datasets import Dataset
    
# -------------------------
# Dataset & Model Preparation
# -------------------------
# def process_func(X_df, y_value=None, config=None):
#     X_df = X_df.sort_index(level="time")
#     id_ = X_df.index.get_level_values("id")[0]
#     seq_len = len(X_df)

#     if y_value is not None:
#         # y_value为True->period就作为label / y_value为False->全部为0
#         label = X_df['period'].values.astype(np.float32) if y_value else np.zeros(seq_len, dtype=np.float32)
#     else:
#         label = None

#     result = {
#         'id': id_,
#         'ts': X_df['value'].values.astype(np.float32),           # shape: [max_length]
#         'label': label
#     }

#     return result

# def create_structured_dataset(X_df, y_series, config=None):
#     grouped = X_df.groupby('id')
#     processed = [
#         process_func(group, y_series.loc[id_], config=config)
#         for id_, group in tqdm(grouped, desc="Processing dataset")
#     ]
#     return Dataset.from_list(processed)

# def collate_fn(batch):
#     ts = [torch.tensor(item['ts'], dtype=torch.float) for item in batch]
#     label = [torch.tensor(item['label'], dtype=torch.float) for item in batch] if 'label' in batch[0] else None
#     id_ = torch.tensor([item['id'] for item in batch], dtype=torch.long)

#     padded_ts = pad_sequence(ts, batch_first=True)  # [B, L] or [B, L, D]
#     padded_label = pad_sequence(label, batch_first=True) if label is not None else None

#     return {
#         'ts': padded_ts.unsqueeze(-1),  # [B, L, nvars=1]
#         'label': padded_label,          # [B, L]
#         'id': id_
#     }

def process_func(X_df, y_value=None, config=None):
    X_df = X_df.sort_index(level="time")
    id_ = X_df.index.get_level_values("id")[0]
    value = X_df['value'].values.astype(np.float32)
    period = X_df['period'].values.astype(np.float32)
    seq_len = len(X_df)

    # ts = [value, period] 作为输入特征
    ts = np.stack([value, period], axis=-1)  # [L, 2]

    # 创建 soft target：基于 period 的变化点生成高斯尖峰
    target = np.zeros(seq_len, dtype=np.float32)
    transition_points = np.where(np.diff(period) != 0)[0]  # 找到period变化点（0->1）

    if y_value == True:
        for idx in transition_points:
            # 在变化点附近创建高斯分布尖峰
            gauss_range = np.arange(seq_len)
            gauss = scipy.stats.norm.pdf(gauss_range, loc=idx + 1, scale=config.spike_std)
            gauss = gauss / gauss.max()  # 归一化为最大值1
            target += gauss.astype(np.float32)

    result = {
        'id': id_,
        'ts': ts,          # [L, 2]
        'target': target,  # [L]
    }
    if y_value is not None:
        result['label'] = y_value

    return result

def create_structured_dataset(X_df, y_series, config=None):
    grouped = X_df.groupby('id')
    processed = [
        process_func(group, y_series.loc[id_], config=config)
        for id_, group in tqdm(grouped, desc="Processing dataset")
    ]
    return Dataset.from_list(processed)

def collate_fn(batch):
    id_ = torch.tensor([item['id'] for item in batch], dtype=torch.long)
    ts = [torch.tensor(item['ts'], dtype=torch.float) for item in batch]
    target = [torch.tensor(item['target'], dtype=torch.float) for item in batch] if 'target' in batch[0] else None
    label = torch.tensor([item['label'] for item in batch], dtype=torch.long) if 'label' in batch[0] else None

    padded_ts = pad_sequence(ts, batch_first=True)  # [B, L, 2]
    padded_target = pad_sequence(target, batch_first=True) if target is not None else None

    return {
        'id': id_,
        'ts': padded_ts,             # [B, L, 2]
        'target': padded_target,     # [B, L]
        'label': label,       # [B]
    }

def plot_series_with_target(ts, target, id_=None):
    """
    ts: ndarray of shape [L, 2] (value, period)
    target: ndarray of shape [L]
    """
    value = ts[:, 0]
    period = ts[:, 1]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # 左侧 y 轴：绘制 value
    ax1.plot(value, label='value', color='tab:blue', linewidth=2)
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Value", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 右侧 y 轴：绘制 period 和 target
    ax2 = ax1.twinx()
    ax2.plot(period, label='period', color='tab:orange', linestyle='--', alpha=0.7)
    ax2.plot(target, label='target (soft label)', color='tab:red', linewidth=2)
    ax2.set_ylabel("Period / Target", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 图例合并
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    if id_ is not None:
        plt.title(f"ID: {id_}")
    else:
        plt.title("Time Series with Soft Target")

    plt.grid(True)
    plt.tight_layout()
    plt.show()


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