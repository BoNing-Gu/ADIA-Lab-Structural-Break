import torch
from pathlib import Path

# --- Dirs ---
# Define project root
ROOT_DIR = Path(__file__).resolve().parent.parent # Go up one level to the project root

# --- Data Dirs ---
DATA_DIR = ROOT_DIR / 'data'

# --- Data files ---
DATA_PATH = '../data/X_train.parquet'
TARGET_PATH = '../data/y_train.parquet'
TEST_DATA_PATH = '../data/X_test.parquet'

# Model hyperparameters
INPUT_DIM = 1
HIDDEN_DIM = 64
N_LAYERS = 2
OUTPUT_DIM = 1
BIDIRECTIONAL = True
DROPOUT = 0.2

# Training configurations
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 10
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# Logging
LOG_DIR = './logs'
MODEL_SAVE_DIR = './saved_models' 