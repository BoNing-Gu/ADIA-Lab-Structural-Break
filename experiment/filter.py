import pandas as pd
import numpy as np
import os
from . import config, features

def parse_feature_name(feature_name):
    """
    解析特征名，提取mode_name和func_id
    特征名格式: f"{mode_name}_{func_id}_{col}"
    其中mode_name中不包含下划线
    
    Returns:
        tuple: (mode_name, func_id, col) 或 (None, None, None) 如果解析失败
    """
    parts = feature_name.split('_')
    if len(parts) >= 3:
        mode_name = parts[0]
        func_id = parts[1]
        col = '_'.join(parts[2:])  # 处理col中可能包含下划线的情况
        return mode_name, func_id, col
    return None, None, None


def check_new_features_corr(feature_df, loaded_features, drop_flag=True, threshold=0.95):
    """
    检查新特征与已加载特征的相关性
    
    Args:
        feature_df: 包含所有特征的DataFrame
        loaded_features: 已加载的特征列表
        drop_flag: 是否删除高相关性特征
        threshold: 相关性阈值
    
    Returns:
        tuple: (处理后的feature_df, 被删除的特征列表)
    """
    new_features = [col for col in feature_df.columns if col not in loaded_features]
    print(f"\n[check_corr] 新特征数量: {len(new_features)}")
    print(f"[check_corr] 已加载特征数量: {len(loaded_features)}")
    
    if not new_features:
        print("[check_corr] 没有新特征需要检查")
        return feature_df, []
    
    # 计算新特征与已加载特征的相关性
    all_features = list(loaded_features) + new_features
    corr_matrix = feature_df[all_features].corr()
    cross_corr = corr_matrix.loc[new_features, loaded_features]
    
    # 检查高相关性特征 (|corr| > 0.7)
    high_corr_features = cross_corr[(cross_corr.abs() > 0.75).any(axis=1)]
    
    if not high_corr_features.empty:
        print("\n[check_corr] 发现高相关性新特征 (|corr| > 0.7):")
        for new_feat in high_corr_features.index:
            correlated_with = high_corr_features.columns[high_corr_features.loc[new_feat].abs() > 0.7]
            corr_values = high_corr_features.loc[new_feat, high_corr_features.loc[new_feat].abs() > 0.7]
            
            print(f"\n  {new_feat} 与以下特征高度相关:")
            for loaded_feat, corr in zip(correlated_with, corr_values):
                print(f"    - {loaded_feat}: {corr:.3f}")
    else:
        print("\n[check_corr] 没有发现高相关性新特征 (|corr| > 0.7)")
    
    # 删除高度相关的新特征（严格大于 threshold）
    dropped_features = []
    if drop_flag:
        high_corr_to_drop = cross_corr[(cross_corr.abs() > threshold).any(axis=1)]
        dropped_features = list(high_corr_to_drop.index)
        if dropped_features:
            print(f"\n[check_corr] 删除 {len(dropped_features)} 个高相关性新特征 (|corr| > {threshold}):")
            for feat in dropped_features:
                print(f"  - {feat}")
            feature_df = feature_df.drop(columns=dropped_features)
        else:
            print(f"\n[check_corr] 没有特征超过阈值 |corr| > {threshold}，无需删除")
    
    return feature_df, dropped_features


def corr_filter(feature_file: str = None, threshold: float = 0.95, drop_flag: bool = True):
    """
    基于特征相关性进行筛选，根据特征命名规则逐类检查相关性
    
    特征命名规则: f"{mode_name}_{func_id}_{col}"
    - 第一类特征（按出现顺序）不进行相关性检查
    - 后续每类特征只与前面所有类的特征进行相关性检查
    - 类内特征不互相检查相关性
    
    Args:
        feature_file (str, optional): 特征文件名。如果未指定，将使用最新版本。
        threshold (float): 相关性阈值，默认0.95
        drop_flag (bool): 是否实际删除高相关性特征，默认False（仅报告）
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
    
    # 3. 按特征命名规则分组
    feature_groups = {}  # {(mode_name, func_id): [feature_names]}
    ungrouped_features = []  # 无法解析的特征
    
    for feature_name in feature_df.columns:
        mode_name, func_id, col = parse_feature_name(feature_name)
        if mode_name is not None and func_id is not None:
            key = (mode_name, func_id)
            if key not in feature_groups:
                feature_groups[key] = []
            feature_groups[key].append(feature_name)
        else:
            ungrouped_features.append(feature_name)
    
    print(f"\n[filter_corr] 特征分组结果:")
    print(f"  - 成功分组: {len(feature_groups)} 个组")
    print(f"  - 无法分组: {len(ungrouped_features)} 个特征")
    
    for i, (key, features_in_group) in enumerate(feature_groups.items()):
        mode_name, func_id = key
        print(f"  组 {i+1}: {mode_name}_{func_id} ({len(features_in_group)} 个特征)")
    
    if ungrouped_features:
        print(f"  未分组特征: {ungrouped_features}")
    
    # 4. 逐类进行相关性检查
    all_dropped_features = []
    processed_features = list(ungrouped_features)  # 从未分组特征开始
    group_keys = list(feature_groups.keys())
    
    print(f"\n[filter_corr] 开始逐类相关性检查 (阈值: {threshold}):")
    
    for i, key in enumerate(group_keys):
        mode_name, func_id = key
        current_group_features = feature_groups[key]
        
        print(f"\n--- 处理第 {i+1} 类特征: {mode_name}_{func_id} ---")
        print(f"当前组特征数量: {len(current_group_features)}")
        print(f"已处理特征数量: {len(processed_features)}")
        
        if i == 0:
            # 第一类特征不进行相关性检查
            print("第一类特征，跳过相关性检查")
            processed_features.extend(current_group_features)
        else:
            # 检查当前组特征与已处理特征的相关性
            temp_df = feature_df[processed_features + current_group_features].copy()
            temp_df, dropped_in_group = check_new_features_corr(
                temp_df, 
                processed_features, 
                drop_flag=drop_flag, 
                threshold=threshold
            )
            
            all_dropped_features.extend(dropped_in_group)
            
            # 更新已处理特征列表（添加未被删除的当前组特征）
            remaining_group_features = [f for f in current_group_features if f not in dropped_in_group]
            processed_features.extend(remaining_group_features)
            
            print(f"当前组保留特征: {len(remaining_group_features)}")
            print(f"当前组删除特征: {len(dropped_in_group)}")
    
    # 5. 应用删除操作（如果启用）
    final_feature_df = feature_df.copy()
    if drop_flag and all_dropped_features:
        final_feature_df = feature_df.drop(columns=all_dropped_features)
        print(f"\n[filter_corr] 总共删除 {len(all_dropped_features)} 个高相关性特征")
    
    keep_features = [f for f in feature_df.columns if f not in all_dropped_features]
    
    # 6. 保存结果到txt文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f'# 特征相关性筛选结果（逐类检查）\n')
        f.write(f'# 特征文件: {loaded_feature_name}\n')
        f.write(f'# 相关性阈值: {threshold}\n')
        f.write(f'# 是否删除: {drop_flag}\n')
        f.write(f'# 原始特征数量: {len(feature_df.columns)}\n')
        f.write(f'# 删除特征数量: {len(all_dropped_features)}\n')
        f.write(f'# 保留特征数量: {len(keep_features)}\n')
        f.write(f'# 特征组数量: {len(feature_groups)}\n')
        f.write(f'# 未分组特征数量: {len(ungrouped_features)}\n')
        f.write('\n' + '=' * 50 + '\n\n')
        
        # 写入特征分组信息
        f.write('# 特征分组信息\n')
        for i, (key, features_in_group) in enumerate(feature_groups.items()):
            mode_name, func_id = key
            f.write(f'# 组 {i+1}: {mode_name}_{func_id} ({len(features_in_group)} 个特征)\n')
        f.write('\n')
        
        if ungrouped_features:
            f.write('# 未分组特征:\n')
            for feat in ungrouped_features:
                f.write(f'#   - {feat}\n')
            f.write('\n')
        
        # 写入需要删除的特征列表
        f.write('# 需要删除的高相关性特征\n')
        f.write('drop_features = [\n')
        for feat in all_dropped_features:
            f.write(f"    '{feat}',\n")
        f.write(']\n\n')
        
        f.write('\n' + '*' * 50 + '\n\n')
        
        # 写入保留的特征列表
        f.write('# 保留的特征\n')
        f.write('keep_features = [\n')
        for feat in keep_features:
            f.write(f"    '{feat}',\n")
        f.write(']\n\n')
        
        # 按组写入特征详情
        f.write('\n' + '-' * 50 + '\n')
        f.write('# 各组特征详情\n\n')
        
        for i, (key, features_in_group) in enumerate(feature_groups.items()):
            mode_name, func_id = key
            group_dropped = [f for f in features_in_group if f in all_dropped_features]
            group_kept = [f for f in features_in_group if f not in all_dropped_features]
            
            f.write(f'# 组 {i+1}: {mode_name}_{func_id}\n')
            f.write(f'# 原始: {len(features_in_group)}, 保留: {len(group_kept)}, 删除: {len(group_dropped)}\n')
            
            if group_kept:
                f.write(f'{mode_name}_{func_id}_kept = [\n')
                for feat in group_kept:
                    f.write(f"    '{feat}',\n")
                f.write(']\n\n')
            
            if group_dropped:
                f.write(f'{mode_name}_{func_id}_dropped = [\n')
                for feat in group_dropped:
                    f.write(f"    '{feat}',\n")
                f.write(']\n\n')
    
    print(f"\n[filter_corr] 相关性筛选结果已保存至 {output_file}")
    print(f"[filter_corr] 最终统计: 原始 {len(feature_df.columns)} -> 保留 {len(keep_features)} (删除 {len(all_dropped_features)})")
    
    return final_feature_df if drop_flag else feature_df, all_dropped_features
    

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

def feature_imp_filter(train_version: str, feature_file: str = None, top_k: list[int] = None):
    if top_k is None:
        top_k = [200, 300, 400, 500]

    # 1. 加载特征数据
    imp_path = os.path.join(config.OUTPUT_DIR, train_version, 'feature_importance.tsv')
    df = pd.read_csv(imp_path, sep='\t')
    _, loaded_feature_name = features.load_features(feature_file, data_ids=["0"])
    
    # 2. 创建输出目录
    feature_name_without_ext = loaded_feature_name.replace('.parquet', '') if loaded_feature_name.endswith('.parquet') else loaded_feature_name
    output_dir = os.path.join(config.OUTPUT_DIR, f'filter_{feature_name_without_ext}')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'filtered_by_feature_imp_{train_version}.txt')

    # 3. 保存结果到txt文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for k in top_k:
            selected_features = (
                df.sort_values('importance', ascending=False)['feature']
                .head(k)
                .tolist()
            )
            f.write(f'# Top {k}\n')
            f.write('top_features = [\n')
            f.writelines([f"    '{feat}',\n" for feat in selected_features])
            f.write(']\n\n')
        f.write('\n' + '*' * 50 + '\n')
        
    print(f"[filter_feature_imp] 特征筛选结果已保存至 {output_file}")