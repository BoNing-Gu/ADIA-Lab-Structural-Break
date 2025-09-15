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
os.environ['OMP_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

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
from tsfresh.feature_extraction import extract_features, EfficientFCParameters, ComprehensiveFCParameters

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
    # try:
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
    # except Exception as e:
    #     print(f"[ERROR] Failed to extract features for id={id_name}: {e}")
    #     return pd.DataFrame()  # 返回空 df 以防止中断

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
    # extracted_features = extract_features(
    #     X_train_flat, 
    #     default_fc_parameters=EfficientFCParameters(),
    #     column_id="id", column_sort="time", n_jobs=72)
    extracted_features = parallel_extract_features(X_train_flat, n_jobs=72)
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
    df_0.sort_index(inplace=True)
    df_1.set_index('raw_id', inplace=True)
    df_1.sort_index(inplace=True)
    df_diff = df_1.subtract(df_0)
    df_diff.sort_index(inplace=True)
    df_0.columns = [f'{col}_left' for col in df_0.columns]
    df_1.columns = [f'{col}_right' for col in df_1.columns]
    df_diff.columns = [f'{col}_diff' for col in df_diff.columns]
    print(f'df_0: {df_0.shape}')
    print(f'df_1: {df_1.shape}')
    print(f'df_diff: {df_diff.shape}')
    
    # 按索引拼接整体特征和差值特征
    df_combined = pd.concat([df_whole, df_diff, df_0, df_1], axis=1)
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
feature_df = clean_feature_names(feature_df)
# impute(feature_df)

# # FDR Filter
# features_filtered = select_features(feature_df, y_train['structural_breakpoint'], n_jobs=0, fdr_level=0.1)
# features_filtered = clean_feature_names(features_filtered)
# with open("./feature_dfs/features-tsfresh_autoextract_fdrfilter.txt", "w", encoding="utf-8") as f:
#     for col in features_filtered.columns:
#         f.write(col + "\n")
# print(f'After FDR filter: {features_filtered.shape}')
# filtered_path = './feature_dfs/features-tsfresh_autoextract+fdrfilter.parquet'
# features_filtered.index.name = "id"
# features_filtered.to_parquet(filtered_path)
# print(f"Saved filtered features: {features_filtered.shape}")


# Corr Filter
columns = ['value__abs_energy_whole',
 'value__absolute_maximum_diff',
 'value__agg_autocorrelation__f_agg__mean___maxlag_40_diff',
 'value__agg_autocorrelation__f_agg__mean___maxlag_40_whole',
 'value__agg_linear_trend__attr__intercept___chunk_len_10__f_agg__max__left',
 'value__agg_linear_trend__attr__intercept___chunk_len_10__f_agg__max__right',
 'value__agg_linear_trend__attr__intercept___chunk_len_10__f_agg__mean__whole',
 'value__agg_linear_trend__attr__intercept___chunk_len_50__f_agg__max__right',
 'value__agg_linear_trend__attr__intercept___chunk_len_50__f_agg__mean__whole',
 'value__agg_linear_trend__attr__intercept___chunk_len_50__f_agg__var__right',
 'value__agg_linear_trend__attr__intercept___chunk_len_5__f_agg__max__right',
 'value__agg_linear_trend__attr__intercept___chunk_len_5__f_agg__mean__diff',
 'value__agg_linear_trend__attr__rvalue___chunk_len_10__f_agg__mean__left',
 'value__agg_linear_trend__attr__rvalue___chunk_len_10__f_agg__mean__whole',
 'value__agg_linear_trend__attr__rvalue___chunk_len_10__f_agg__var__left',
 'value__agg_linear_trend__attr__rvalue___chunk_len_50__f_agg__max__diff',
 'value__agg_linear_trend__attr__rvalue___chunk_len_50__f_agg__max__left',
 'value__agg_linear_trend__attr__rvalue___chunk_len_50__f_agg__mean__diff',
 'value__agg_linear_trend__attr__rvalue___chunk_len_50__f_agg__mean__left',
 'value__agg_linear_trend__attr__rvalue___chunk_len_50__f_agg__var__left',
 'value__agg_linear_trend__attr__rvalue___chunk_len_50__f_agg__var__right',
 'value__agg_linear_trend__attr__rvalue___chunk_len_5__f_agg__max__diff',
 'value__agg_linear_trend__attr__rvalue___chunk_len_5__f_agg__mean__diff',
 'value__agg_linear_trend__attr__rvalue___chunk_len_5__f_agg__min__left',
 'value__agg_linear_trend__attr__rvalue___chunk_len_5__f_agg__var__left',
 'value__agg_linear_trend__attr__slope___chunk_len_10__f_agg__max__left',
 'value__agg_linear_trend__attr__slope___chunk_len_10__f_agg__min__left',
 'value__agg_linear_trend__attr__slope___chunk_len_50__f_agg__max__diff',
 'value__agg_linear_trend__attr__slope___chunk_len_50__f_agg__max__left',
 'value__agg_linear_trend__attr__slope___chunk_len_50__f_agg__min__left',
 'value__agg_linear_trend__attr__slope___chunk_len_5__f_agg__min__left',
 'value__agg_linear_trend__attr__slope___chunk_len_5__f_agg__var__right',
 'value__agg_linear_trend__attr__stderr___chunk_len_10__f_agg__var__diff',
 'value__agg_linear_trend__attr__stderr___chunk_len_10__f_agg__var__right',
 'value__agg_linear_trend__attr__stderr___chunk_len_50__f_agg__max__diff',
 'value__agg_linear_trend__attr__stderr___chunk_len_50__f_agg__mean__left',
 'value__agg_linear_trend__attr__stderr___chunk_len_50__f_agg__min__right',
 'value__agg_linear_trend__attr__stderr___chunk_len_50__f_agg__var__left',
 'value__agg_linear_trend__attr__stderr___chunk_len_5__f_agg__max__right',
 'value__agg_linear_trend__attr__stderr___chunk_len_5__f_agg__mean__left',
 'value__agg_linear_trend__attr__stderr___chunk_len_5__f_agg__var__left',
 'value__ar_coefficient__coeff_0__k_10_right',
 'value__ar_coefficient__coeff_0__k_10_whole',
 'value__ar_coefficient__coeff_10__k_10_left',
 'value__ar_coefficient__coeff_10__k_10_whole',
 'value__ar_coefficient__coeff_2__k_10_diff',
 'value__ar_coefficient__coeff_2__k_10_left',
 'value__ar_coefficient__coeff_3__k_10_right',
 'value__ar_coefficient__coeff_5__k_10_left',
 'value__ar_coefficient__coeff_8__k_10_diff',
 'value__augmented_dickey_fuller__attr__usedlag___autolag__AIC__diff',
 'value__autocorrelation__lag_2_diff',
 'value__autocorrelation__lag_2_left',
 'value__autocorrelation__lag_2_right',
 'value__autocorrelation__lag_3_diff',
 'value__autocorrelation__lag_6_right',
 'value__autocorrelation__lag_7_whole',
 'value__benford_correlation_left',
 'value__benford_correlation_whole',
 'value__binned_entropy__max_bins_10_whole',
 'value__change_quantiles__f_agg__mean___isabs_False__qh_0_2__ql_0_0_right',
 'value__change_quantiles__f_agg__mean___isabs_False__qh_0_4__ql_0_0_right',
 'value__change_quantiles__f_agg__mean___isabs_False__qh_0_6__ql_0_4_left',
 'value__change_quantiles__f_agg__mean___isabs_False__qh_0_6__ql_0_4_whole',
 'value__change_quantiles__f_agg__mean___isabs_False__qh_0_8__ql_0_2_right',
 'value__change_quantiles__f_agg__mean___isabs_False__qh_1_0__ql_0_2_diff',
 'value__change_quantiles__f_agg__mean___isabs_False__qh_1_0__ql_0_2_left',
 'value__change_quantiles__f_agg__mean___isabs_False__qh_1_0__ql_0_4_left',
 'value__change_quantiles__f_agg__mean___isabs_False__qh_1_0__ql_0_4_whole',
 'value__change_quantiles__f_agg__mean___isabs_False__qh_1_0__ql_0_6_diff',
 'value__change_quantiles__f_agg__mean___isabs_False__qh_1_0__ql_0_6_whole',
 'value__change_quantiles__f_agg__mean___isabs_False__qh_1_0__ql_0_8_left',
 'value__change_quantiles__f_agg__mean___isabs_False__qh_1_0__ql_0_8_right',
 'value__change_quantiles__f_agg__mean___isabs_True__qh_0_2__ql_0_0_left',
 'value__change_quantiles__f_agg__mean___isabs_True__qh_0_4__ql_0_0_left',
 'value__change_quantiles__f_agg__mean___isabs_True__qh_0_4__ql_0_2_whole',
 'value__change_quantiles__f_agg__mean___isabs_True__qh_0_6__ql_0_0_diff',
 'value__change_quantiles__f_agg__mean___isabs_True__qh_0_6__ql_0_0_whole',
 'value__change_quantiles__f_agg__mean___isabs_True__qh_0_6__ql_0_4_right',
 'value__change_quantiles__f_agg__mean___isabs_True__qh_0_6__ql_0_4_whole',
 'value__change_quantiles__f_agg__mean___isabs_True__qh_0_8__ql_0_2_diff',
 'value__change_quantiles__f_agg__mean___isabs_True__qh_0_8__ql_0_4_whole',
 'value__change_quantiles__f_agg__mean___isabs_True__qh_1_0__ql_0_0_diff',
 'value__change_quantiles__f_agg__mean___isabs_True__qh_1_0__ql_0_2_diff',
 'value__change_quantiles__f_agg__mean___isabs_True__qh_1_0__ql_0_4_diff',
 'value__change_quantiles__f_agg__mean___isabs_True__qh_1_0__ql_0_4_left',
 'value__change_quantiles__f_agg__mean___isabs_True__qh_1_0__ql_0_6_diff',
 'value__change_quantiles__f_agg__mean___isabs_True__qh_1_0__ql_0_6_left',
 'value__change_quantiles__f_agg__mean___isabs_True__qh_1_0__ql_0_8_left',
 'value__change_quantiles__f_agg__var___isabs_False__qh_0_2__ql_0_0_right',
 'value__change_quantiles__f_agg__var___isabs_False__qh_0_4__ql_0_0_whole',
 'value__change_quantiles__f_agg__var___isabs_False__qh_0_6__ql_0_0_whole',
 'value__change_quantiles__f_agg__var___isabs_False__qh_0_6__ql_0_4_diff',
 'value__change_quantiles__f_agg__var___isabs_False__qh_0_6__ql_0_4_whole',
 'value__change_quantiles__f_agg__var___isabs_False__qh_1_0__ql_0_2_left',
 'value__change_quantiles__f_agg__var___isabs_False__qh_1_0__ql_0_6_left',
 'value__change_quantiles__f_agg__var___isabs_False__qh_1_0__ql_0_6_right',
 'value__change_quantiles__f_agg__var___isabs_False__qh_1_0__ql_0_8_right',
 'value__change_quantiles__f_agg__var___isabs_True__qh_0_2__ql_0_0_right',
 'value__change_quantiles__f_agg__var___isabs_True__qh_0_4__ql_0_0_right',
 'value__change_quantiles__f_agg__var___isabs_True__qh_0_4__ql_0_0_whole',
 'value__change_quantiles__f_agg__var___isabs_True__qh_0_6__ql_0_0_whole',
 'value__change_quantiles__f_agg__var___isabs_True__qh_0_6__ql_0_4_whole',
 'value__change_quantiles__f_agg__var___isabs_True__qh_0_8__ql_0_0_whole',
 'value__change_quantiles__f_agg__var___isabs_True__qh_0_8__ql_0_2_whole',
 'value__change_quantiles__f_agg__var___isabs_True__qh_1_0__ql_0_2_left',
 'value__cid_ce__normalize_False_left',
 'value__cid_ce__normalize_False_whole',
 'value__count_above__t_0_diff',
 'value__count_above__t_0_left',
 'value__count_above__t_0_right',
 'value__count_above__t_0_whole',
 'value__count_below__t_0_diff',
 'value__count_below__t_0_left',
 'value__count_below__t_0_right',
 'value__cwt_coefficients__coeff_0__w_10__widths__2__5__10__20__right',
 'value__cwt_coefficients__coeff_0__w_2__widths__2__5__10__20__right',
 'value__cwt_coefficients__coeff_11__w_10__widths__2__5__10__20__diff',
 'value__cwt_coefficients__coeff_12__w_5__widths__2__5__10__20__diff',
 'value__cwt_coefficients__coeff_14__w_5__widths__2__5__10__20__whole',
 'value__cwt_coefficients__coeff_1__w_5__widths__2__5__10__20__right',
 'value__cwt_coefficients__coeff_2__w_5__widths__2__5__10__20__right',
 'value__cwt_coefficients__coeff_3__w_20__widths__2__5__10__20__diff',
 'value__cwt_coefficients__coeff_3__w_5__widths__2__5__10__20__diff',
 'value__energy_ratio_by_chunks__num_segments_10__segment_focus_0_right',
 'value__energy_ratio_by_chunks__num_segments_10__segment_focus_1_right',
 'value__energy_ratio_by_chunks__num_segments_10__segment_focus_2_right',
 'value__energy_ratio_by_chunks__num_segments_10__segment_focus_4_left',
 'value__energy_ratio_by_chunks__num_segments_10__segment_focus_4_whole',
 'value__energy_ratio_by_chunks__num_segments_10__segment_focus_5_diff',
 'value__energy_ratio_by_chunks__num_segments_10__segment_focus_7_left',
 'value__energy_ratio_by_chunks__num_segments_10__segment_focus_7_right',
 'value__energy_ratio_by_chunks__num_segments_10__segment_focus_9_left',
 'value__fft_aggregated__aggtype__skew__right',
 'value__fft_aggregated__aggtype__skew__whole',
 'value__fft_coefficient__attr__abs___coeff_0_diff',
 'value__fft_coefficient__attr__abs___coeff_0_whole',
 'value__fft_coefficient__attr__abs___coeff_10_diff',
 'value__fft_coefficient__attr__abs___coeff_17_left',
 'value__fft_coefficient__attr__abs___coeff_1_left',
 'value__fft_coefficient__attr__abs___coeff_1_right',
 'value__fft_coefficient__attr__abs___coeff_1_whole',
 'value__fft_coefficient__attr__abs___coeff_22_left',
 'value__fft_coefficient__attr__abs___coeff_24_whole',
 'value__fft_coefficient__attr__abs___coeff_29_right',
 'value__fft_coefficient__attr__abs___coeff_2_left',
 'value__fft_coefficient__attr__abs___coeff_34_diff',
 'value__fft_coefficient__attr__abs___coeff_47_right',
 'value__fft_coefficient__attr__abs___coeff_54_diff',
 'value__fft_coefficient__attr__abs___coeff_54_whole',
 'value__fft_coefficient__attr__abs___coeff_56_diff',
 'value__fft_coefficient__attr__abs___coeff_5_diff',
 'value__fft_coefficient__attr__abs___coeff_62_left',
 'value__fft_coefficient__attr__abs___coeff_68_diff',
 'value__fft_coefficient__attr__abs___coeff_69_diff',
 'value__fft_coefficient__attr__abs___coeff_69_whole',
 'value__fft_coefficient__attr__abs___coeff_6_diff',
 'value__fft_coefficient__attr__abs___coeff_71_right',
 'value__fft_coefficient__attr__abs___coeff_74_whole',
 'value__fft_coefficient__attr__abs___coeff_79_right',
 'value__fft_coefficient__attr__abs___coeff_80_diff',
 'value__fft_coefficient__attr__abs___coeff_82_diff',
 'value__fft_coefficient__attr__abs___coeff_82_left',
 'value__fft_coefficient__attr__abs___coeff_83_left',
 'value__fft_coefficient__attr__abs___coeff_90_whole',
 'value__fft_coefficient__attr__abs___coeff_96_left',
 'value__fft_coefficient__attr__abs___coeff_99_right',
 'value__fft_coefficient__attr__angle___coeff_10_left',
 'value__fft_coefficient__attr__angle___coeff_11_diff',
 'value__fft_coefficient__attr__angle___coeff_12_right',
 'value__fft_coefficient__attr__angle___coeff_17_left',
 'value__fft_coefficient__attr__angle___coeff_17_right',
 'value__fft_coefficient__attr__angle___coeff_17_whole',
 'value__fft_coefficient__attr__angle___coeff_18_right',
 'value__fft_coefficient__attr__angle___coeff_19_diff',
 'value__fft_coefficient__attr__angle___coeff_1_whole',
 'value__fft_coefficient__attr__angle___coeff_20_left',
 'value__fft_coefficient__attr__angle___coeff_21_diff',
 'value__fft_coefficient__attr__angle___coeff_22_whole',
 'value__fft_coefficient__attr__angle___coeff_23_diff',
 'value__fft_coefficient__attr__angle___coeff_23_left',
 'value__fft_coefficient__attr__angle___coeff_23_whole',
 'value__fft_coefficient__attr__angle___coeff_24_diff',
 'value__fft_coefficient__attr__angle___coeff_26_whole',
 'value__fft_coefficient__attr__angle___coeff_27_whole',
 'value__fft_coefficient__attr__angle___coeff_29_diff',
 'value__fft_coefficient__attr__angle___coeff_29_left',
 'value__fft_coefficient__attr__angle___coeff_31_whole',
 'value__fft_coefficient__attr__angle___coeff_34_diff',
 'value__fft_coefficient__attr__angle___coeff_34_right',
 'value__fft_coefficient__attr__angle___coeff_37_diff',
 'value__fft_coefficient__attr__angle___coeff_37_left',
 'value__fft_coefficient__attr__angle___coeff_37_right',
 'value__fft_coefficient__attr__angle___coeff_38_diff',
 'value__fft_coefficient__attr__angle___coeff_38_right',
 'value__fft_coefficient__attr__angle___coeff_39_right',
 'value__fft_coefficient__attr__angle___coeff_41_right',
 'value__fft_coefficient__attr__angle___coeff_42_whole',
 'value__fft_coefficient__attr__angle___coeff_44_left',
 'value__fft_coefficient__attr__angle___coeff_44_right',
 'value__fft_coefficient__attr__angle___coeff_47_whole',
 'value__fft_coefficient__attr__angle___coeff_49_left',
 'value__fft_coefficient__attr__angle___coeff_50_right',
 'value__fft_coefficient__attr__angle___coeff_54_left',
 'value__fft_coefficient__attr__angle___coeff_55_diff',
 'value__fft_coefficient__attr__angle___coeff_55_right',
 'value__fft_coefficient__attr__angle___coeff_56_diff',
 'value__fft_coefficient__attr__angle___coeff_59_left',
 'value__fft_coefficient__attr__angle___coeff_59_right',
 'value__fft_coefficient__attr__angle___coeff_61_whole',
 'value__fft_coefficient__attr__angle___coeff_62_diff',
 'value__fft_coefficient__attr__angle___coeff_70_whole',
 'value__fft_coefficient__attr__angle___coeff_71_diff',
 'value__fft_coefficient__attr__angle___coeff_71_left',
 'value__fft_coefficient__attr__angle___coeff_75_diff',
 'value__fft_coefficient__attr__angle___coeff_75_left',
 'value__fft_coefficient__attr__angle___coeff_75_right',
 'value__fft_coefficient__attr__angle___coeff_75_whole',
 'value__fft_coefficient__attr__angle___coeff_77_whole',
 'value__fft_coefficient__attr__angle___coeff_79_diff',
 'value__fft_coefficient__attr__angle___coeff_79_left',
 'value__fft_coefficient__attr__angle___coeff_79_right',
 'value__fft_coefficient__attr__angle___coeff_79_whole',
 'value__fft_coefficient__attr__angle___coeff_80_diff',
 'value__fft_coefficient__attr__angle___coeff_81_diff',
 'value__fft_coefficient__attr__angle___coeff_81_whole',
 'value__fft_coefficient__attr__angle___coeff_82_right',
 'value__fft_coefficient__attr__angle___coeff_82_whole',
 'value__fft_coefficient__attr__angle___coeff_83_diff',
 'value__fft_coefficient__attr__angle___coeff_85_diff',
 'value__fft_coefficient__attr__angle___coeff_85_right',
 'value__fft_coefficient__attr__angle___coeff_86_right',
 'value__fft_coefficient__attr__angle___coeff_87_whole',
 'value__fft_coefficient__attr__angle___coeff_9_whole',
 'value__fft_coefficient__attr__imag___coeff_1_whole',
 'value__fft_coefficient__attr__imag___coeff_20_left',
 'value__fft_coefficient__attr__imag___coeff_21_whole',
 'value__fft_coefficient__attr__imag___coeff_24_left',
 'value__fft_coefficient__attr__imag___coeff_24_right',
 'value__fft_coefficient__attr__imag___coeff_28_right',
 'value__fft_coefficient__attr__imag___coeff_29_left',
 'value__fft_coefficient__attr__imag___coeff_2_diff',
 'value__fft_coefficient__attr__imag___coeff_2_left',
 'value__fft_coefficient__attr__imag___coeff_32_diff',
 'value__fft_coefficient__attr__imag___coeff_32_right',
 'value__fft_coefficient__attr__imag___coeff_33_diff',
 'value__fft_coefficient__attr__imag___coeff_33_right',
 'value__fft_coefficient__attr__imag___coeff_3_left',
 'value__fft_coefficient__attr__imag___coeff_42_whole',
 'value__fft_coefficient__attr__imag___coeff_44_right',
 'value__fft_coefficient__attr__imag___coeff_45_right',
 'value__fft_coefficient__attr__imag___coeff_46_whole',
 'value__fft_coefficient__attr__imag___coeff_47_right',
 'value__fft_coefficient__attr__imag___coeff_48_diff',
 'value__fft_coefficient__attr__imag___coeff_4_left',
 'value__fft_coefficient__attr__imag___coeff_4_whole',
 'value__fft_coefficient__attr__imag___coeff_50_diff',
 'value__fft_coefficient__attr__imag___coeff_50_left',
 'value__fft_coefficient__attr__imag___coeff_52_diff',
 'value__fft_coefficient__attr__imag___coeff_53_left',
 'value__fft_coefficient__attr__imag___coeff_54_left',
 'value__fft_coefficient__attr__imag___coeff_55_diff',
 'value__fft_coefficient__attr__imag___coeff_55_right',
 'value__fft_coefficient__attr__imag___coeff_57_diff',
 'value__fft_coefficient__attr__imag___coeff_57_whole',
 'value__fft_coefficient__attr__imag___coeff_5_whole',
 'value__fft_coefficient__attr__imag___coeff_60_diff',
 'value__fft_coefficient__attr__imag___coeff_60_left',
 'value__fft_coefficient__attr__imag___coeff_62_left',
 'value__fft_coefficient__attr__imag___coeff_62_right',
 'value__fft_coefficient__attr__imag___coeff_65_diff',
 'value__fft_coefficient__attr__imag___coeff_66_left',
 'value__fft_coefficient__attr__imag___coeff_69_whole',
 'value__fft_coefficient__attr__imag___coeff_6_whole',
 'value__fft_coefficient__attr__imag___coeff_70_left',
 'value__fft_coefficient__attr__imag___coeff_72_diff',
 'value__fft_coefficient__attr__imag___coeff_72_left',
 'value__fft_coefficient__attr__imag___coeff_73_diff',
 'value__fft_coefficient__attr__imag___coeff_74_right',
 'value__fft_coefficient__attr__imag___coeff_75_whole',
 'value__fft_coefficient__attr__imag___coeff_77_whole',
 'value__fft_coefficient__attr__imag___coeff_87_right',
 'value__fft_coefficient__attr__imag___coeff_8_diff',
 'value__fft_coefficient__attr__imag___coeff_8_whole',
 'value__fft_coefficient__attr__imag___coeff_90_diff',
 'value__fft_coefficient__attr__imag___coeff_90_right',
 'value__fft_coefficient__attr__imag___coeff_96_whole',
 'value__fft_coefficient__attr__imag___coeff_9_left',
 'value__fft_coefficient__attr__real___coeff_11_diff',
 'value__fft_coefficient__attr__real___coeff_13_diff',
 'value__fft_coefficient__attr__real___coeff_15_left',
 'value__fft_coefficient__attr__real___coeff_16_left',
 'value__fft_coefficient__attr__real___coeff_17_right',
 'value__fft_coefficient__attr__real___coeff_18_right',
 'value__fft_coefficient__attr__real___coeff_23_whole',
 'value__fft_coefficient__attr__real___coeff_26_left',
 'value__fft_coefficient__attr__real___coeff_26_right',
 'value__fft_coefficient__attr__real___coeff_27_left',
 'value__fft_coefficient__attr__real___coeff_27_whole',
 'value__fft_coefficient__attr__real___coeff_2_whole',
 'value__fft_coefficient__attr__real___coeff_30_diff',
 'value__fft_coefficient__attr__real___coeff_33_whole',
 'value__fft_coefficient__attr__real___coeff_37_right',
 'value__fft_coefficient__attr__real___coeff_40_left',
 'value__fft_coefficient__attr__real___coeff_40_whole',
 'value__fft_coefficient__attr__real___coeff_46_diff',
 'value__fft_coefficient__attr__real___coeff_47_whole',
 'value__fft_coefficient__attr__real___coeff_48_whole',
 'value__fft_coefficient__attr__real___coeff_4_diff',
 'value__fft_coefficient__attr__real___coeff_51_right',
 'value__fft_coefficient__attr__real___coeff_52_left',
 'value__fft_coefficient__attr__real___coeff_53_left',
 'value__fft_coefficient__attr__real___coeff_53_whole',
 'value__fft_coefficient__attr__real___coeff_54_diff',
 'value__fft_coefficient__attr__real___coeff_54_right',
 'value__fft_coefficient__attr__real___coeff_55_right',
 'value__fft_coefficient__attr__real___coeff_57_whole',
 'value__fft_coefficient__attr__real___coeff_5_left',
 'value__fft_coefficient__attr__real___coeff_60_diff',
 'value__fft_coefficient__attr__real___coeff_60_whole',
 'value__fft_coefficient__attr__real___coeff_61_diff',
 'value__fft_coefficient__attr__real___coeff_66_diff',
 'value__fft_coefficient__attr__real___coeff_72_right',
 'value__fft_coefficient__attr__real___coeff_74_right',
 'value__fft_coefficient__attr__real___coeff_75_left',
 'value__fft_coefficient__attr__real___coeff_75_right',
 'value__fft_coefficient__attr__real___coeff_77_whole',
 'value__fft_coefficient__attr__real___coeff_78_diff',
 'value__fft_coefficient__attr__real___coeff_78_left',
 'value__fft_coefficient__attr__real___coeff_7_right',
 'value__fft_coefficient__attr__real___coeff_83_left',
 'value__fft_coefficient__attr__real___coeff_83_right',
 'value__fft_coefficient__attr__real___coeff_84_right',
 'value__fft_coefficient__attr__real___coeff_86_diff',
 'value__fft_coefficient__attr__real___coeff_87_left',
 'value__fft_coefficient__attr__real___coeff_87_right',
 'value__fft_coefficient__attr__real___coeff_93_diff',
 'value__fft_coefficient__attr__real___coeff_94_right',
 'value__fft_coefficient__attr__real___coeff_97_right',
 'value__first_location_of_maximum_left',
 'value__fourier_entropy__bins_10_whole',
 'value__friedrich_coefficients__coeff_1__m_3__r_30_whole',
 'value__friedrich_coefficients__coeff_2__m_3__r_30_diff',
 'value__friedrich_coefficients__coeff_3__m_3__r_30_diff',
 'value__friedrich_coefficients__coeff_3__m_3__r_30_left',
 'value__friedrich_coefficients__coeff_3__m_3__r_30_right',
 'value__friedrich_coefficients__coeff_3__m_3__r_30_whole',
 'value__has_duplicate_whole',
 'value__index_mass_quantile__q_0_1_diff',
 'value__index_mass_quantile__q_0_1_right',
 'value__index_mass_quantile__q_0_2_right',
 'value__index_mass_quantile__q_0_3_right',
 'value__index_mass_quantile__q_0_4_right',
 'value__index_mass_quantile__q_0_6_left',
 'value__index_mass_quantile__q_0_6_right',
 'value__index_mass_quantile__q_0_6_whole',
 'value__index_mass_quantile__q_0_7_left',
 'value__index_mass_quantile__q_0_7_whole',
 'value__index_mass_quantile__q_0_8_left',
 'value__index_mass_quantile__q_0_9_left',
 'value__index_mass_quantile__q_0_9_whole',
 'value__kurtosis_left',
 'value__kurtosis_right',
 'value__last_location_of_maximum_diff',
 'value__last_location_of_maximum_left',
 'value__last_location_of_minimum_left',
 'value__last_location_of_minimum_right',
 'value__lempel_ziv_complexity__bins_100_left',
 'value__lempel_ziv_complexity__bins_10_whole',
 'value__lempel_ziv_complexity__bins_3_left',
 'value__lempel_ziv_complexity__bins_5_whole',
 'value__linear_trend__attr__intercept__diff',
 'value__linear_trend__attr__pvalue__whole',
 'value__linear_trend__attr__rvalue__diff',
 'value__linear_trend__attr__rvalue__whole',
 'value__maximum_whole',
 'value__mean_abs_change_diff',
 'value__mean_change_diff',
 'value__mean_change_left',
 'value__mean_diff',
 'value__mean_left',
 'value__mean_n_absolute_max__number_of_maxima_7_left',
 'value__mean_right',
 'value__mean_second_derivative_central_left',
 'value__mean_whole',
 'value__median_diff',
 'value__median_left',
 'value__median_right',
 'value__median_whole',
 'value__number_cwt_peaks__n_1_diff',
 'value__number_peaks__n_50_diff',
 'value__number_peaks__n_5_left',
 'value__partial_autocorrelation__lag_5_diff',
 'value__partial_autocorrelation__lag_9_left',
 'value__percentage_of_reoccurring_datapoints_to_all_datapoints_diff',
 'value__percentage_of_reoccurring_datapoints_to_all_datapoints_left',
 'value__percentage_of_reoccurring_datapoints_to_all_datapoints_right',
 'value__percentage_of_reoccurring_datapoints_to_all_datapoints_whole',
 'value__percentage_of_reoccurring_values_to_all_values_diff',
 'value__percentage_of_reoccurring_values_to_all_values_left',
 'value__percentage_of_reoccurring_values_to_all_values_right',
 'value__percentage_of_reoccurring_values_to_all_values_whole',
 'value__permutation_entropy__dimension_3__tau_1_left',
 'value__permutation_entropy__dimension_5__tau_1_left',
 'value__permutation_entropy__dimension_6__tau_1_whole',
 'value__quantile__q_0_1_diff',
 'value__quantile__q_0_1_whole',
 'value__quantile__q_0_2_diff',
 'value__quantile__q_0_2_whole',
 'value__quantile__q_0_3_right',
 'value__quantile__q_0_4_diff',
 'value__quantile__q_0_4_right',
 'value__quantile__q_0_4_whole',
 'value__quantile__q_0_6_diff',
 'value__quantile__q_0_7_diff',
 'value__quantile__q_0_7_right',
 'value__quantile__q_0_9_left',
 'value__ratio_beyond_r_sigma__r_0_5_diff',
 'value__ratio_beyond_r_sigma__r_0_5_left',
 'value__ratio_beyond_r_sigma__r_1_5_diff',
 'value__ratio_beyond_r_sigma__r_1_5_whole',
 'value__ratio_beyond_r_sigma__r_1_left',
 'value__ratio_beyond_r_sigma__r_1_right',
 'value__ratio_beyond_r_sigma__r_2_5_right',
 'value__ratio_beyond_r_sigma__r_2_5_whole',
 'value__ratio_beyond_r_sigma__r_2_diff',
 'value__ratio_beyond_r_sigma__r_2_left',
 'value__ratio_beyond_r_sigma__r_2_whole',
 'value__ratio_beyond_r_sigma__r_3_left',
 'value__ratio_beyond_r_sigma__r_3_whole',
 'value__ratio_value_number_to_time_series_length_diff',
 'value__ratio_value_number_to_time_series_length_right',
 'value__ratio_value_number_to_time_series_length_whole',
 'value__skewness_right',
 'value__spkt_welch_density__coeff_8_whole',
 'value__standard_deviation_diff',
 'value__standard_deviation_whole',
 'value__sum_of_reoccurring_data_points_diff',
 'value__sum_of_reoccurring_data_points_whole',
 'value__sum_of_reoccurring_values_whole',
 'value__sum_values_diff',
 'value__sum_values_whole',
 'value__time_reversal_asymmetry_statistic__lag_1_left',
 'value__variance_diff',
 'value__variance_whole',
 'value__variation_coefficient_diff',
 'value__variation_coefficient_whole']
loaded_feature_df = pd.read_parquet('feature_dfs/backups/features_20250819_233429_id_0.parquet')
print(f'Load: {loaded_feature_df.shape}')
feature_df = feature_df[columns]
feature_df = pd.concat([loaded_feature_df, feature_df], axis=1)
feature_df, removed_features = check_new_features_corr(feature_df, loaded_feature_df, drop_flag=True, threshold=0.95)
with open("./feature_dfs/features-tsfresh_autoextract_corrfilter.txt", "w", encoding="utf-8") as f:
    for col in feature_df.columns:
        if col not in loaded_feature_df.columns:
            f.write(col + "\n")
feature_df = feature_df[[col for col in feature_df.columns if col not in loaded_feature_df.columns]]
print(f'After corr filter: {feature_df.shape}')


# # Perm Filter
# print("Starting 80-20 train-test split...")
# feature_importances = pd.DataFrame(index=feature_df.columns)
# permutation_results = pd.DataFrame(index=feature_df.columns)
# perm_imp = True
# train_idx, val_idx = train_test_split(
#     range(len(feature_df)),
#     test_size=0.2,
#     random_state=42,
#     stratify=y_train
# )

# # --- Model ---
# LGBM_PARAMS = {
#     # --- 基础设定 ---
#     'objective': 'binary',
#     'metric': 'auc',
#     'boosting_type': 'gbdt',
#     'n_estimators': 3000, 
#     'learning_rate': 0.005,
#     'num_leaves': 29,
#     'random_state': 42,
#     'n_jobs': 64,

#     # --- 正则化和采样 ---
#     'reg_alpha': 3,          # L1 正则化
#     'reg_lambda': 3,         # L2 正则化
#     'colsample_bytree': 0.8,   # 构建树时对特征的列采样率
#     'subsample': 0.8,          # 训练样本的采样率
# }

# start_time = time.time()

# X_train_split, y_train_split = feature_df.iloc[train_idx], y_train.iloc[train_idx]
# X_val_split, y_val_split = feature_df.iloc[val_idx], y_train.iloc[val_idx]

# model = lgb.LGBMClassifier(**LGBM_PARAMS)

# model.fit(
#     X_train_split, y_train_split,
#     eval_set=[(X_train_split, y_train_split), (X_val_split, y_val_split)],
#     eval_names=['train', 'valid'],
#     eval_metric='auc',
#     # callbacks=[lgb.early_stopping(100, verbose=False)]
# )

# preds = model.predict_proba(X_val_split)[:, 1]
# feature_importances['importance'] = model.feature_importances_

# train_auc = model.best_score_['train']['auc']
# val_auc = roc_auc_score(y_val_split, preds)
# print(f"Train AUC: {train_auc:.5f}, Val AUC: {val_auc:.5f}")

# duration = time.time() - start_time
# print(f"Training finished in {duration:.2f}s")

# # 计算permutation importance
# if perm_imp:
#     print("Calculating permutation importance...")
#     perm_start_time = time.time()
#     # 在验证集上计算permutation importance
#     perm_result = permutation_importance(
#         model, X_val_split, y_val_split,
#         n_repeats=20,  # 可以根据需要调整重复次数
#         random_state=42,
#         scoring='roc_auc',
#         n_jobs=72
#     )
#     # 保存permutation importance结果
#     permutation_results['importance'] = perm_result.importances_mean
#     perm_duration = time.time() - perm_start_time
#     print(f"Permutation importance finished in {perm_duration:.2f}s")

# # 保存permutation importance
# if perm_imp:
#     feature_names = feature_df.columns
#     df = pd.DataFrame({
#         'feature': feature_names,
#         'permutation_importance': permutation_results['importance'],
#     })
#     df = df.sort_values('permutation_importance', ascending=False)
    
#     save_path = "./feature_dfs/features-tsfresh_autoextract_permfilter.tsv"
#     df.to_csv(save_path, sep='\t', index=False)