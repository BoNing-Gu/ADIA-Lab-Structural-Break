import pandas as pd
import numpy as np
import scipy.stats
import statsmodels as sm
import statsmodels.tsa.api as tsa
import antropy

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import os
import re
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# 1. read dataset
print("Loading data...")
X_train = pd.read_parquet('../data/X_train.parquet')
y_train = pd.read_parquet('../data/y_train.parquet')
print("Data loaded.")


# 特征提取
from tsfresh.feature_extraction import extract_features, EfficientFCParameters

def reshape_for_tsfresh(X_train: pd.DataFrame) -> pd.DataFrame:
    # Reset index to access 'id' and 'time'
    df = X_train.reset_index()
    # 构建 flat 格式
    df['id'] = df['id'].astype(str) + "_" + df['period'].astype(str)
    df = df[['id', 'time', 'value']]
    df['time'] = df.groupby('id').cumcount()
    df.sort_values(by=['id', 'time'], inplace=True)

    return df

def extract_single_id_features(id_group: pd.DataFrame, id_name: str) -> pd.DataFrame:
    try:
        df = extract_features(
            id_group,
            default_fc_parameters=EfficientFCParameters(),
            column_id="id",
            column_sort="time",
            column_value="value",
            n_jobs=0,
            disable_progressbar=True,
        )
        df.index = [id_name]  # 确保 index 是 id 字符串
        return df
    except Exception as e:
        print(f"[ERROR] Failed to extract features for id={id_name}: {e}")
        return pd.DataFrame()  # 返回空 df 以防止中断

def parallel_extract_features(X_flat: pd.DataFrame, n_jobs: int = 8) -> pd.DataFrame:
    # 分组为 dict：id -> df
    grouped = dict(tuple(X_flat.groupby('id')))
    ids = list(grouped.keys())

    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_single_id_features)(grouped[id_], id_)
        for id_ in tqdm(ids, desc="Extracting tsfresh features")
    )

    # 拼接结果
    features_df = pd.concat(results, axis=0)
    return features_df

extracted_path = '../feature_dfs/features-tsfresh_autoextract.parquet'
if os.path.exists(extracted_path):
    extracted_features = pd.read_parquet(extracted_path)
    print(f"Loaded existing extracted features: {extracted_features.shape}")
else:
    print("Extracting features...")
    flat_path = '../data/X_train_flat.parquet'
    if os.path.exists(flat_path):
        X_train_flat = pd.read_parquet(flat_path)
        print(f"X_train_flat is loaded. {X_train_flat.shape}")
    else:
        X_train_flat = reshape_for_tsfresh(X_train)
        X_train_flat.to_parquet(flat_path)
        print(f"X_train_flat is created and saved. {X_train_flat.shape}")
    extracted_features = extract_features(
        X_train_flat, 
        default_fc_parameters=EfficientFCParameters(),
        column_id="id", column_sort="time", n_jobs=0)
    # extracted_features = parallel_extract_features(X_train_flat, n_jobs=8)
    extracted_features.to_parquet(extracted_path)
    print(f"Saved extracted features: {extracted_features.shape}")


# 特征筛选
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

def compute_period_diff_features(extracted_features: pd.DataFrame) -> pd.DataFrame:
    df = extracted_features.copy().reset_index()
    df[['raw_id', 'period']] = df['index'].str.extract(r'^(.*)_(\d)$')
    df['raw_id'] = df['raw_id'].astype(int)
    df['period'] = df['period'].astype(int)

    # 分 period
    df_0 = df[df['period'] == 0].drop(columns=['period', 'index'])
    df_1 = df[df['period'] == 1].drop(columns=['period', 'index'])

    # 设置 raw_id 为索引
    df_0.set_index('raw_id', inplace=True)
    df_1.set_index('raw_id', inplace=True)

    # 差值
    diff_df = df_1.subtract(df_0)
    diff_df.sort_index(inplace=True)
    return diff_df

def clean_feature_names(df: pd.DataFrame, prefix: str = "f") -> pd.DataFrame:
    cleaned_columns = []
    for i, col in enumerate(df.columns):
        # 替换非法字符为 _
        cleaned = re.sub(r'[^\w]', '_', col)
        # 防止开头是数字（如 "123_feature"）非法
        if re.match(r'^\d', cleaned):
            cleaned = f"{prefix}_{cleaned}"
        # 多个连续 _ 合并为一个
        cleaned = re.sub(r'__+', '_', cleaned)
        cleaned_columns.append(cleaned)
    df.columns = cleaned_columns
    return df

filtered_path = '../feature_dfs/features-tsfresh_autoextract+filter.parquet'
if os.path.exists(filtered_path):
    features_filtered = pd.read_parquet(filtered_path)
    print(f"Loaded existing filtered features: {features_filtered.shape}")
else:
    diff_features = compute_period_diff_features(extracted_features)
    print(f"Diff extracted features: {diff_features.shape}")
    impute(diff_features)
    features_filtered = select_features(diff_features, y_train['structural_breakpoint'])
    features_filtered.index.name = "id"
    features_filtered = clean_feature_names(features_filtered)
    features_filtered.to_parquet(filtered_path)
    print(f"Saved filtered features: {features_filtered.shape}")