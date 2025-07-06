import pandas as pd
import logging
from . import config

logger = logging.getLogger(__name__)

def load_data():
    """加载训练数据"""
    try:
        logger.info("Loading data...")
        X_train = pd.read_parquet(config.TRAIN_X_FILE)
        y_train = pd.read_parquet(config.TRAIN_Y_FILE)
        logger.info("Data loaded successfully.")
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        return X_train, y_train
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        raise 