import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

# --- Dirs ---
# 定义项目根目录
ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent

# --- Data & Feature Dirs ---
DATA_DIR = PROJECT_ROOT / 'data'
FEATURE_DIR = PROJECT_ROOT / 'feature_dfs'
FEATURE_BACKUP_DIR = FEATURE_DIR / 'backups'

# --- Output & Log Dirs ---
OUTPUT_DIR = PROJECT_ROOT / 'output'
LOG_DIR = ROOT_DIR / 'logs'

# --- Files ---
TRAIN_X_FILE = DATA_DIR / 'X_train.parquet'
TRAIN_Y_FILE = DATA_DIR / 'y_train.parquet'
FEATURE_FILE = FEATURE_DIR / 'features.parquet' # 持久化的特征文件
LOG_FILE = LOG_DIR / 'experiment.log'


# --- Model ---
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'random_state': 42,
    'n_estimators': 1000, 
    'learning_rate': 0.005,
    'num_leaves': 31,
    'n_jobs': -1,
}

# --- CV ---
CV_PARAMS = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42
} 