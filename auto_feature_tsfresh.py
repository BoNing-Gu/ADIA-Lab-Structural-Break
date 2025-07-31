import pandas as pd
import numpy as np
import scipy.stats
import statsmodels as sm
import statsmodels.tsa.api as tsa
import time

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

import os
import re
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

print("Loading data...")
X_train = pd.read_parquet('./data/X_train.parquet')
y_train = pd.read_parquet('./data/y_train.parquet')
print("Data loaded.")


# 特征提取
from tsfresh.feature_extraction import extract_features, EfficientFCParameters

def reshape_for_tsfresh(X_train: pd.DataFrame) -> pd.DataFrame:
    print(f'X_train: {X_train.shape}')
    # Reset index to access 'id' and 'time'
    df = X_train.reset_index()
    df_whole = df.copy()
    df_whole['id'] = df_whole['id'].astype(str)
    # 构建 flat 格式
    df['id'] = df['id'].astype(str) + "_" + df['period'].astype(str)
    df = pd.concat([df, df_whole], axis=0)
    df = df[['id', 'time', 'value']]
    df['time'] = df.groupby('id').cumcount()
    df.sort_values(by=['id', 'time'], inplace=True)
    print(f'X_train_flat: {df.shape}')
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

extracted_path = './feature_dfs/features-tsfresh_autoextract.parquet'
if os.path.exists(extracted_path):
    extracted_features = pd.read_parquet(extracted_path)
    print(f"Loaded existing extracted features: {extracted_features.shape}")
else:
    print("Extracting features...")
    flat_path = './data/X_train_flat.parquet'
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
    print(extracted_features.head())


# 特征筛选
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import train_test_split

def compute_features(extracted_features: pd.DataFrame) -> pd.DataFrame:
    df = extracted_features.copy().reset_index()
    
    # 提取整体特征
    df_whole = df[~df['index'].str.contains('_')]
    df_whole['raw_id'] = df_whole['index'].astype(int)
    df_whole = df_whole.drop(columns=['index'])
    df_whole.set_index('raw_id', inplace=True)
    df_whole.sort_index(inplace=True)
    df_whole.columns = [f'{col}_whole' for col in df_whole.columns]
    print(f'df_whole: {df_whole.shape}')

    # 提取分段特征差值
    df = df[df['index'].str.contains('_')]
    df[['raw_id', 'period']] = df['index'].str.extract(r'^(.*)_(\d)$')
    df['raw_id'] = df['raw_id'].astype(int)
    df['period'] = df['period'].astype(int)
    df_0 = df[df['period'] == 0].drop(columns=['period', 'index'])
    df_1 = df[df['period'] == 1].drop(columns=['period', 'index'])
    df_0.set_index('raw_id', inplace=True)
    df_1.set_index('raw_id', inplace=True)
    df_diff = df_1.subtract(df_0)
    df_diff.sort_index(inplace=True)
    df_diff.columns = [f'{col}_diff' for col in df_diff.columns]
    print(f'df_diff: {df_diff.shape}')
    
    # 按索引拼接整体特征和差值特征
    df_combined = pd.concat([df_whole, df_diff], axis=1)
    print(f'df_combined: {df_combined.shape}')
    return df_combined

def clean_feature_names(df: pd.DataFrame, prefix: str = "f") -> pd.DataFrame:
    cleaned_columns = []
    for i, col in enumerate(df.columns):
        # 替换非法字符为 _
        cleaned = re.sub(r'[^\w]', '_', col)
        # 防止开头是数字（如 "123_feature"）非法
        if re.match(r'^\d', cleaned):
            cleaned = f"{prefix}_{cleaned}"
        # 多个连续 _ 合并为一个
        # cleaned = re.sub(r'__+', '_', cleaned)
        cleaned_columns.append(cleaned)
    df.columns = cleaned_columns
    return df


# 剔除高相关特征
def check_new_features_corr(feature_df, loaded_feature_df, drop_flag=False, threshold=0.95):
    """检查新特征与已加载特征的相关性"""
    new_features = [col for col in feature_df.columns if col not in loaded_feature_df.columns]
    loaded_features = loaded_feature_df.columns
    print(f"\nNumber of new features: {len(new_features)}")
    print(f"Number of loaded features: {len(loaded_features)}")
    
    # 计算新特征与已加载特征的相关性
    corr_matrix = feature_df[new_features + list(loaded_features)].corr()
    cross_corr = corr_matrix.loc[new_features, loaded_features]
    high_corr_features = cross_corr[(cross_corr.abs() > 0.7).any(axis=1)]
    
    if not high_corr_features.empty:
        print("\nNew features with high correlation (|corr| > 0.7) to loaded features:")
        # 打印每个高相关性新特征及其相关特征
        for new_feat in high_corr_features.index:
            correlated_with = high_corr_features.columns[high_corr_features.loc[new_feat].abs() > 0.7]
            corr_values = high_corr_features.loc[new_feat, high_corr_features.loc[new_feat].abs() > 0.7]
            
            print(f"\n{new_feat} is highly correlated with:")
            for loaded_feat, corr in zip(correlated_with, corr_values):
                print(f"  - {loaded_feat}: {corr:.3f}")
    else:
        print("\nNo new features show high correlation (|corr| > 0.7) with loaded features.")
        
    # 删除高度相关的新特征（严格大于 threshold）
    dropped_features = []
    if drop_flag:
        high_corr_to_drop = cross_corr[(cross_corr.abs() > threshold).any(axis=1)]
        dropped_features = list(high_corr_to_drop.index)
        if dropped_features:
            print(f"\nDropping {len(dropped_features)} new features with |corr| > {threshold}:")
            for feat in dropped_features:
                print(f"  - {feat}")
            feature_df = feature_df.drop(columns=dropped_features)
        else:
            print(f"\nNo new features exceeded threshold |corr| > {threshold}, nothing dropped.")

    return feature_df, dropped_features


feature_df = compute_features(extracted_features)
impute(feature_df)


# FDR Filter
features_filtered = select_features(feature_df, y_train['structural_breakpoint'], n_jobs=0, fdr_level=0.1)
features_filtered = clean_feature_names(features_filtered)
with open("./feature_dfs/features-tsfresh_autoextract_fdrfilter.txt", "w", encoding="utf-8") as f:
    for col in features_filtered.columns:
        f.write(col + "\n")
print(f'After FDR filter: {features_filtered.shape}')
# filtered_path = './feature_dfs/features-tsfresh_autoextract+fdrfilter.parquet'
# features_filtered.index.name = "id"
# features_filtered.to_parquet(filtered_path)
# print(f"Saved filtered features: {features_filtered.shape}")


# Corr Filter
loaded_feature_df = pd.read_parquet('feature_dfs/features_20250730_182640_id_0.parquet')
print(f'Load: {loaded_feature_df.shape}')
feature_df = pd.concat([loaded_feature_df, feature_df], axis=1)
feature_df, removed_features = check_new_features_corr(feature_df, loaded_feature_df, drop_flag=True, threshold=0.95)
with open("./feature_dfs/features-tsfresh_autoextract_corrfilter.txt", "w", encoding="utf-8") as f:
    for col in feature_df.columns:
        if col not in loaded_feature_df.columns:
            f.write(col + "\n")
feature_df = feature_df[[col for col in feature_df.columns if col not in loaded_feature_df.columns]]
print(f'After corr filter: {feature_df.shape}')


# Perm Filter
print("Starting 80-20 train-test split...")
feature_df = clean_feature_names(feature_df)
feature_importances = pd.DataFrame(index=feature_df.columns)
permutation_results = pd.DataFrame(index=feature_df.columns)
perm_imp = True
train_idx, val_idx = train_test_split(
    range(len(feature_df)),
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# --- Model ---
LGBM_PARAMS = {
    # --- 基础设定 ---
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 4000, 
    'learning_rate': 0.005,
    'num_leaves': 29,
    'random_state': 42,
    'n_jobs': 12,

    # --- 正则化和采样 ---
    'reg_alpha': 3,          # L1 正则化
    'reg_lambda': 3,         # L2 正则化
    'colsample_bytree': 0.8,   # 构建树时对特征的列采样率
    'subsample': 0.8,          # 训练样本的采样率
}

start_time = time.time()

X_train_split, y_train_split = feature_df.iloc[train_idx], y_train.iloc[train_idx]
X_val_split, y_val_split = feature_df.iloc[val_idx], y_train.iloc[val_idx]

model = lgb.LGBMClassifier(**LGBM_PARAMS)

model.fit(
    X_train_split, y_train_split,
    eval_set=[(X_train_split, y_train_split), (X_val_split, y_val_split)],
    eval_names=['train', 'valid'],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

preds = model.predict_proba(X_val_split)[:, 1]
feature_importances['importance'] = model.feature_importances_

train_auc = model.best_score_['train']['auc']
val_auc = roc_auc_score(y_val_split, preds)
print(f"Train AUC: {train_auc:.5f}, Val AUC: {val_auc:.5f}")

duration = time.time() - start_time
print(f"Training finished in {duration:.2f}s")

# 计算permutation importance
if perm_imp:
    print("Calculating permutation importance...")
    perm_start_time = time.time()
    # 在验证集上计算permutation importance
    perm_result = permutation_importance(
        model, X_val_split, y_val_split,
        n_repeats=20,  # 可以根据需要调整重复次数
        random_state=42,
        scoring='roc_auc',
        n_jobs=12
    )
    # 保存permutation importance结果
    permutation_results['importance'] = perm_result.importances_mean
    perm_duration = time.time() - perm_start_time
    print(f"Permutation importance finished in {perm_duration:.2f}s")

# 保存permutation importance
if perm_imp:
    feature_names = feature_df.columns
    df = pd.DataFrame({
        'feature': feature_names,
        'permutation_importance': permutation_results['importance'],
    })
    df = df.sort_values('permutation_importance', ascending=False)
    
    save_path = "./feature_dfs/features-tsfresh_autoextract_permfilter.tsv"
    df.to_csv(save_path, sep='\t', index=False)