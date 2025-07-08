import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from pathlib import Path
import config

class TimeSeriesDataset(Dataset):
    """
    Dataset class for the structural break data.
    Correctly handles loading data from parquet files and splitting by 'period'.
    """
    def __init__(self, df, labels_df=None, is_test=False):
        self.df = df
        # Create a mapping from series_id to its group of data
        self.grouped = self.df.groupby('series_id')
        self.series_ids = list(self.grouped.groups.keys())
        
        self.labels_df = labels_df
        self.is_test = is_test

    def __len__(self):
        return len(self.series_ids)

    def __getitem__(self, idx):
        series_id = self.series_ids[idx]
        series_data = self.grouped.get_group(series_id)
        
        # Split data into two phases based on the 'period' column
        phase1_data = series_data[series_data['period'] == 0]['value'].values.astype(np.float32)
        phase2_data = series_data[series_data['period'] == 1]['value'].values.astype(np.float32)

        phase1_tensor = torch.from_numpy(phase1_data).unsqueeze(-1)
        phase2_tensor = torch.from_numpy(phase2_data).unsqueeze(-1)
        
        if self.is_test:
            return series_id, phase1_tensor, phase2_tensor
        else:
            label = self.labels_df.loc[series_id, 'structural_breakpoint']
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return phase1_tensor, phase2_tensor, label_tensor

def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    It pads sequences to the max length in each batch.
    """
    # Unzip the batch
    phase1_tensors, phase2_tensors, labels = zip(*batch)

    # Get sequence lengths
    phase1_lengths = torch.tensor([len(t) for t in phase1_tensors])
    phase2_lengths = torch.tensor([len(t) for t in phase2_tensors])

    # Pad sequences
    phase1_padded = pad_sequence(phase1_tensors, batch_first=True, padding_value=0.0)
    phase2_padded = pad_sequence(phase2_tensors, batch_first=True, padding_value=0.0)

    # Stack labels
    labels = torch.stack(labels)
    
    return (phase1_padded, phase1_lengths), (phase2_padded, phase2_lengths), labels

def get_dataloaders():
    # Construct paths relative to the project root
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / 'data'
    
    # Load data
    X_train_df = pd.read_parquet(data_dir / 'X_train.parquet')
    y_train_df = pd.read_parquet(data_dir / 'y_train.parquet')
    
    # The 'id' level of the MultiIndex is the series_id
    X_train_df = X_train_df.reset_index().rename(columns={'id': 'series_id'})

    # Split data by series_id for stratification
    all_series_ids = y_train_df.index
    train_ids, val_ids = train_test_split(all_series_ids, test_size=config.VAL_SPLIT, random_state=config.RANDOM_SEED, stratify=y_train_df)
    
    # Filter main dataframe for train and validation sets
    train_data_df = X_train_df[X_train_df['series_id'].isin(train_ids)]
    val_data_df = X_train_df[X_train_df['series_id'].isin(val_ids)]

    # Create datasets
    train_dataset = TimeSeriesDataset(train_data_df, y_train_df)
    val_dataset = TimeSeriesDataset(val_data_df, y_train_df)

    # Create dataloaders with the custom collate function
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader

def get_test_dataloader():
    current_dir = Path(__file__).parent
    test_df = pd.read_parquet(current_dir / config.TEST_DATA_PATH)
    test_dataset = TimeSeriesDataset(test_df, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    return test_loader

if __name__ == '__main__':
    train_loader, val_loader = get_dataloaders()
    
    print("Testing DataLoader...")
    # Test one batch
    for (phase1_padded, phase1_lengths), (phase2_padded, phase2_lengths), labels in train_loader:
        print("\n--- Batch Details ---")
        print("Phase 1 Padded Shape:", phase1_padded.shape)
        print("Phase 1 Lengths:", phase1_lengths)
        print("Phase 2 Padded Shape:", phase2_padded.shape)
        print("Phase 2 Lengths:", phase2_lengths)
        print("Labels Shape:", labels.shape)
        print("Labels:", labels)
        
        # Verify that max length in lengths tensor matches padded shape
        assert phase1_padded.shape[1] == phase1_lengths.max().item()
        assert phase2_padded.shape[1] == phase2_lengths.max().item()
        break
    print("\nDataLoader test passed!")

    test_loader = get_test_dataloader()
    # Test one batch from the test_loader
    for batch in test_loader:
        series_ids, phase1, phase2 = batch
        print("\nTest Batch Shapes:")
        print("Series IDs:", len(series_ids))
        print("Phase 1:", phase1.shape)
        print("Phase 2:", phase2.shape)
        break 