import os
import csv
import json
from typing import Any, Dict, Optional
from datetime import datetime
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities import rank_zero_only

# -------------------------
# Logger & Callback Module
# -------------------------
def serialize_value(val: Any) -> Any:
    """
    将不可直接序列化的对象转换成字符串，比如 torch.device，pathlib.Path 等
    """
    import torch
    if isinstance(val, (str, int, float, bool, type(None))):
        return val
    elif isinstance(val, (list, tuple)):
        return [serialize_value(x) for x in val]
    elif isinstance(val, dict):
        return {k: serialize_value(v) for k, v in val.items()}
    elif isinstance(val, torch.device):
        return str(val)
    else:
        # fallback
        return str(val)
        
class CustomLogger(Logger):
    def __init__(self, save_dir, version=None, resume=False):
        super().__init__()
        self._save_dir = save_dir
        self._version = version if version is not None else datetime.now().strftime("%m-%d_%H-%M")
        self.trainer = None
        self._resume = resume
        
        # Initialize file paths
        self._train_file = os.path.join(self.save_dir, self.version, "train_metrics.csv")
        self._val_file = os.path.join(self.save_dir, self.version, "val_metrics.csv")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self._train_file), exist_ok=True)
        os.makedirs(os.path.dirname(self._val_file), exist_ok=True)
        
        # Initialize CSV files with headers if they don't exist
        self._init_csv_files()

    @property
    def name(self):
        return "custom"
        
    @property
    def version(self):
        return self._version

    @property
    def save_dir(self):
        return self._save_dir
    
    def set_trainer(self, trainer):
        self.trainer = trainer

    @rank_zero_only
    def _init_csv_files(self):
        if not self._resume:
            # 若 resume=False，无条件重写文件
            with open(self._train_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "epoch", "train_loss", "train_collapse_loss", "train_total_loss", "time_per_step", "timestamp", "level"])
            with open(self._val_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "val_auc", "timestamp"])
        else:
            # 若 resume=True，只在文件不存在时写入表头
            if not os.path.exists(self._train_file):
                with open(self._train_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["step", "epoch", "train_loss", "train_collapse_loss", "train_total_loss", "time_per_step", "timestamp", "level"])
            if not os.path.exists(self._val_file):
                with open(self._val_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["epoch", "val_auc", "timestamp"])

    @rank_zero_only
    def log_hyperparams(self, params, ignore=None):
        try:
            # Save hyperparameters to a separate JSON file
            ignore = ignore or []
            hp_file = os.path.join(self.save_dir, self.version, "hparams.json")
            os.makedirs(os.path.dirname(hp_file), exist_ok=True)
            # [Debug] 打印原始参数信息
            if isinstance(params, dict):
                param_dict = params
            else:
                param_dict = vars(params)
            # print(f"[Debug] Total params: {len(param_dict)}")
            # print(f"[Debug] Param keys: {list(param_dict.keys())}")
            # for key, value in param_dict.items():
            #     print(f"[Debug] {key}: type={type(value)}")
            filtered_params = {
                key: value
                for key, value in param_dict.items()
                if key not in ignore
            }
            with open(hp_file, 'w') as f:
                json.dump(filtered_params, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[Debug] Failed to log hyperparameters: {e}")
            print(f'[Debug] {params}')

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # Determine if it's training or validation metrics
        if "train_loss_step" in metrics:
            # Training metrics (step-level)
            with open(self._train_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    step,
                    metrics.get("epoch", "N/A"),
                    metrics.get("train_loss_step", "N/A"),
                    metrics.get("train_collapse_loss_step", "N/A"),
                    metrics.get("train_total_loss_step", "N/A"),
                    metrics.get("time_per_step", "N/A"),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "(step)"
                ])
        elif "train_loss_epoch" in metrics:
            # Training metrics (epoch-level)
            with open(self._train_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    step,
                    metrics.get("epoch", "N/A"),
                    metrics.get("train_loss_epoch", "N/A"),
                    metrics.get("train_collapse_loss_epoch", "N/A"),
                    metrics.get("train_total_loss_epoch", "N/A"),
                    "N/A",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "(epoch)"
                ])
        elif "val_auc" in metrics:
            # Validation metrics (epoch-level)
            with open(self._val_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    metrics.get("epoch", "N/A"),
                    metrics.get("val_auc", "N/A"),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])

    @rank_zero_only
    def save(self):
        # Nothing special to save here, as we write to CSV immediately
        pass

    @rank_zero_only
    def finalize(self, status):
        # Nothing special to finalize
        pass
