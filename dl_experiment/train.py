import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import datetime

import config
from model import SiameseLSTM
from data import get_dataloaders

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_outputs = []
    
    for (phase1_padded, phase1_lengths), (phase2_padded, phase2_lengths), labels in tqdm(dataloader, desc="Training"):
        # Move data to device
        phase1_padded, phase1_lengths = phase1_padded.to(device), phase1_lengths.to(device)
        phase2_padded, phase2_lengths = phase2_padded.to(device), phase2_lengths.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model((phase1_padded, phase1_lengths), (phase2_padded, phase2_lengths))
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * phase1_padded.size(0)
        
        all_labels.extend(labels.cpu().numpy())
        all_outputs.extend(outputs.detach().cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_auc = roc_auc_score(all_labels, all_outputs)
    
    return epoch_loss, epoch_auc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for (phase1_padded, phase1_lengths), (phase2_padded, phase2_lengths), labels in tqdm(dataloader, desc="Validating"):
            # Move data to device
            phase1_padded, phase1_lengths = phase1_padded.to(device), phase1_lengths.to(device)
            phase2_padded, phase2_lengths = phase2_padded.to(device), phase2_lengths.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model((phase1_padded, phase1_lengths), (phase2_padded, phase2_lengths))
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * phase1_padded.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.detach().cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_auc = roc_auc_score(all_labels, all_outputs)
    
    return epoch_loss, epoch_auc

def train():
    # Setup logging and saving directories
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    # Dataloaders
    train_loader, val_loader = get_dataloaders()
    
    # Model, criterion, optimizer
    model = SiameseLSTM().to(config.DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_val_auc = 0.0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
        
        val_loss, val_auc = validate_epoch(model, val_loader, criterion, config.DEVICE)
        print(f"Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = os.path.join(config.MODEL_SAVE_DIR, f"best_model_{timestamp}_auc_{val_auc:.4f}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    train() 