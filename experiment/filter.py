import pandas as pd
import numpy as np
import os
from . import config, features

def corr_filter(feature_file: str = None, threshold: float = 0.95):
    """
    基于特征相关性进行筛选，移除高相关性特征
    
    Args:
        feature_file (str, optional): 特征文件名。如果未指定，将使用最新版本。
        threshold (float): 相关性阈值，默认0.95
    """
    # 1. 加载特征数据
    feature_df, loaded_feature_name = features.load_features(feature_file, data_ids=["0"])
    if feature_df is None:
        print(f"[filter_corr] 无法加载特征文件: {feature_file}")
        return
    print(f"[filter_corr] 成功加载特征文件: {loaded_feature_name}")
    print(f"[filter_corr] 特征数量: {len(feature_df.columns)}, 样本数量: {len(feature_df)}")
    
    # 2. 创建输出目录
    feature_name_without_ext = loaded_feature_name.replace('.parquet', '') if loaded_feature_name.endswith('.parquet') else loaded_feature_name
    output_dir = os.path.join(config.OUTPUT_DIR, f'filter_{feature_name_without_ext}')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'filtered_by_correlation.txt')
    
    # 3. 计算相关性
    corr_matrix = feature_df.corr().abs()  # 使用绝对值
    drop_features = []
    feature_names = list(feature_df.columns)
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            feat_i = feature_names[i]
            feat_j = feature_names[j]
            if corr_matrix.iloc[i, j] > threshold and feat_j not in drop_features:
                drop_features.append(feat_j)
                print(f"[filter_corr] 发现高相关性: {feat_i} <-> {feat_j} (相关性: {corr_matrix.iloc[i, j]:.4f})")
    keep_features = [feat for feat in feature_names if feat not in drop_features]
    
    # 4. 保存结果到txt文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f'# 特征相关性筛选结果\n')
        f.write(f'# 特征文件: {loaded_feature_name}\n')
        f.write(f'# 相关性阈值: {threshold}\n')
        f.write(f'# 原始特征数量: {len(feature_names)}\n')
        f.write(f'# 删除特征数量: {len(drop_features)}\n')
        f.write(f'# 保留特征数量: {len(keep_features)}\n')
        f.write('\n' + '=' * 50 + '\n\n')
        
        # 写入需要删除的特征列表
        f.write('# 需要删除的高相关性特征\n')
        f.write('drop_features = [\n')
        for feat in drop_features:
            f.write(f"    '{feat}',\n")
        f.write(']\n\n')
        
        f.write('\n' + '*' * 50 + '\n\n')
        
        # 写入保留的特征列表
        f.write('# 保留的特征\n')
        f.write('keep_features = [\n')
        for feat in keep_features:
            f.write(f"    '{feat}',\n")
        f.write(']\n\n')
    
    print(f"[filter_corr] 相关性筛选结果已保存至 {output_file}")
    

def perm_imp_filter(train_version: str, feature_file: str = None, top_k: list[int] = None, thresholds: list[float] = None):
    if top_k is None:
        top_k = [5, 10, 15]
    if thresholds is None:
        thresholds = [0.0005, 0.0004, 0.0003, 0.0002, 0.0001]

    # 1. 加载特征数据
    imp_path = os.path.join(config.OUTPUT_DIR, train_version, 'permutation_importance.tsv')
    df = pd.read_csv(imp_path, sep='\t')
    _, loaded_feature_name = features.load_features(feature_file, data_ids=["0"])
    
    # 2. 创建输出目录
    feature_name_without_ext = loaded_feature_name.replace('.parquet', '') if loaded_feature_name.endswith('.parquet') else loaded_feature_name
    output_dir = os.path.join(config.OUTPUT_DIR, f'filter_{feature_name_without_ext}')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'filtered_by_perm_imp_{train_version}.txt')

    # 3. 保存结果到txt文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for k in top_k:
            selected_features = (
                df.sort_values('permutation_importance_mean', ascending=False)['feature']
                .head(k)
                .tolist()
            )
            f.write(f'# Top {k}\n')
            f.write('top_features = [\n')
            f.writelines([f"    '{feat}',\n" for feat in selected_features])
            f.write(']\n\n')
        f.write('\n' + '*' * 50 + '\n')
        for th in thresholds:
            selected_features = (
                df.loc[df['permutation_importance_mean'] > th]
                .sort_values('permutation_importance_mean', ascending=False)['feature']
                .tolist()
            )
            f.write(f'# Threshold: {th}\n')
            f.write(f'# Feature Num: {len(selected_features)}\n')
            f.write('filtered_features = [\n')
            f.writelines([f"    '{feat}',\n" for feat in selected_features])
            f.write(']\n\n')

    print(f"[filter_perm_imp] 特征筛选结果已保存至 {output_file}")
