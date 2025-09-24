import pandas as pd
import numpy as np
import os
from . import config, features

def _intra_group_decorrelate(feature_df: pd.DataFrame,
                             group_features: list,
                             threshold: float,
                             report_threshold: float = 0.75):
    """
    在单个组内部进行去相关（贪心保留顺序靠前的特征）。

    返回:
        kept_features, dropped_features
    """
    if not group_features or len(group_features) <= 1:
        return list(group_features), []

    corr_grp = feature_df[group_features].corr()
    kept = []
    dropped = []

    # 展示：报告阈值下的高相关对（可选）
    abs_corr_grp = corr_grp.abs()
    mask_report = (abs_corr_grp > report_threshold) & (~np.eye(len(group_features), dtype=bool))
    if mask_report.any().any():
        print("[intra] 发现组内高相关对 (|corr| > {:.2f})".format(report_threshold))

    for feat in group_features:
        if kept:
            # 与已保留特征的最大相关
            max_with_kept = abs_corr_grp.loc[feat, kept].max()
            if max_with_kept > threshold:
                dropped.append(feat)
                continue
        kept.append(feat)

    return kept, dropped

def _blockwise_cross_corr_flags(feature_df: pd.DataFrame,
                                loaded_features: list,
                                new_features: list,
                                drop_threshold: float,
                                report_threshold: float = 0.75,
                                block_size: int = 512,
                                dtype=np.float32):
    """
    按需分块计算“新特征 × 已加载特征”的交叉相关性，避免构建全量 D×D 相关矩阵。

    仅返回：
      - 需要删除的高相关新特征集合（任一与已加载特征的 |corr| > drop_threshold）
      - 用于展示的高相关对（|corr| > report_threshold）

    说明：
      - 使用 z-score 标准化（ddof=1），相关性通过 Z^T Z / (n-1) 得到
      - 对 NaN 采用均值填充（标准化后等价于 0），常数列视为 std=1（Z 全为 0）
    """
    if not new_features or not loaded_features:
        return set(), {}

    n = len(feature_df)
    if n <= 1:
        return set(), {}

    # 预取均值与标准差（ddof=1），仅对需要的列
    required_cols = list(dict.fromkeys(list(loaded_features) + list(new_features)))
    col_means = feature_df[required_cols].mean(axis=0)
    col_stds = feature_df[required_cols].std(axis=0, ddof=1)

    # 避免除以 0：将 0 标准差替换为 1，使标准化后为 0
    col_stds_replaced = col_stds.replace(0.0, 1.0)

    dropped_new_feats = set()
    report_pairs = {}  # new_feat -> list[(loaded_feat, corr)]

    # 分块遍历新特征与已加载特征
    for new_start in range(0, len(new_features), block_size):
        new_block_feats = new_features[new_start:new_start + block_size]
        X_new = feature_df[new_block_feats].to_numpy(dtype=dtype, copy=False)
        # 标准化: (X - mean) / std
        means_new = col_means[new_block_feats].to_numpy(dtype=dtype)
        stds_new = col_stds_replaced[new_block_feats].to_numpy(dtype=dtype)
        Z_new = (X_new - means_new) / stds_new
        # NaN -> 0（等价于均值填充后标准化）
        np.nan_to_num(Z_new, copy=False)

        # 针对当前 new_block，跟随 processed 分块相乘
        # 记录该 new_block 内每列是否已确定需要删除（提前剪枝）
        new_block_drop_flags = np.zeros(Z_new.shape[1], dtype=bool)

        for proc_start in range(0, len(loaded_features), block_size):
            proc_block_feats = loaded_features[proc_start:proc_start + block_size]
            X_proc = feature_df[proc_block_feats].to_numpy(dtype=dtype, copy=False)
            means_proc = col_means[proc_block_feats].to_numpy(dtype=dtype)
            stds_proc = col_stds_replaced[proc_block_feats].to_numpy(dtype=dtype)
            Z_proc = (X_proc - means_proc) / stds_proc
            np.nan_to_num(Z_proc, copy=False)

            # 计算交叉相关块: (bp x bc)
            # 注意 Z_new: (n x bc)，Z_proc: (n x bp)
            # 目标: Z_proc.T @ Z_new / (n-1)
            corr_block = (Z_proc.T @ Z_new) / (n - 1)

            # 展示用：记录 |corr| > report_threshold 的对
            abs_corr_block = np.abs(corr_block)
            mask_report = abs_corr_block > report_threshold
            if mask_report.any():
                proc_idx, new_idx = np.where(mask_report)
                for pi, ni in zip(proc_idx, new_idx):
                    new_feat = new_block_feats[ni]
                    proc_feat = proc_block_feats[pi]
                    corr_val = float(corr_block[pi, ni])
                    if new_feat not in report_pairs:
                        report_pairs[new_feat] = []
                    report_pairs[new_feat].append((proc_feat, corr_val))

            # 删除判定：任一已加载特征与该新特征 |corr| > drop_threshold
            # 若某列已标记删除，可跳过其后续比较（剪枝）
            if drop_threshold is not None and drop_threshold < 1.0:
                # 针对未标记为删除的列
                remaining_mask = ~new_block_drop_flags
                if remaining_mask.any():
                    # 仅对这些列求列向量的最大绝对相关
                    max_abs_over_proc = abs_corr_block[:, remaining_mask].max(axis=0)
                    to_drop_local = max_abs_over_proc > drop_threshold
                    if to_drop_local.any():
                        # 标记这些列
                        remaining_indices = np.where(remaining_mask)[0]
                        drop_indices = remaining_indices[to_drop_local]
                        new_block_drop_flags[drop_indices] = True

            # 若该 new_block 所有列均已标记删除，可提前结束该 proc 循环
            if new_block_drop_flags.all():
                break

        # 将标记为删除的列名加入集合
        for idx, flagged in enumerate(new_block_drop_flags):
            if flagged:
                dropped_new_feats.add(new_block_feats[idx])

    return dropped_new_feats, report_pairs

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


def check_new_features_corr(feature_df, loaded_features, drop_flag=True, threshold=0.95, corr_matrix=None, use_blockwise: bool = False, block_size: int = 512, new_features_list=None):
    """
    检查新特征与已加载特征的相关性
    
    Args:
        feature_df: 包含所有特征的DataFrame
        loaded_features: 已加载的特征列表
        drop_flag: 是否删除高相关性特征
        threshold: 相关性阈值
        corr_matrix: 预计算的相关性矩阵，如果提供则直接使用，否则重新计算
        use_blockwise: 是否使用分块交叉相关计算（避免全量矩阵）
        block_size: 分块大小
        new_features_list: 显式指定“新特征”列名列表；若为 None，则用差集推断
    
    Returns:
        tuple: (处理后的feature_df, 被删除的特征列表)
    """
    # 推断或使用显式的新特征列表
    if new_features_list is not None:
        new_features = list(new_features_list)
    else:
        new_features = [col for col in feature_df.columns if col not in loaded_features]
    print(f"\n[check_corr] 新特征数量: {len(new_features)}")
    print(f"[check_corr] 已加载特征数量: {len(loaded_features)}")
    
    if not new_features:
        print("[check_corr] 没有新特征需要检查")
        return feature_df, []
    
    dropped_features = []

    if corr_matrix is not None:
        print("[check_corr] 使用预计算的相关性矩阵")
        cross_corr = corr_matrix.loc[new_features, loaded_features]

        # 展示高相关性（报告阈值 0.75）
        high_corr_features = cross_corr[(cross_corr.abs() > 0.75).any(axis=1)]
        if not high_corr_features.empty:
            print("\n[check_corr] 发现高相关性新特征 (|corr| > 0.75):")
            for new_feat in high_corr_features.index:
                mask = high_corr_features.loc[new_feat].abs() > 0.75
                correlated_with = high_corr_features.columns[mask]
                corr_values = high_corr_features.loc[new_feat, mask]
                print(f"\n  {new_feat} 与以下特征高度相关:")
                for loaded_feat, corr in zip(correlated_with, corr_values):
                    print(f"    - {loaded_feat}: {corr:.3f}")
        else:
            print("\n[check_corr] 没有发现高相关性新特征 (|corr| > 0.75)")

        # 删除高度相关的新特征（严格大于 threshold）
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

    # 分块路径：避免构建全量相关矩阵
    if use_blockwise:
        print("[check_corr] 使用分块交叉相关（避免全量矩阵）")
        dropped_set, report_pairs = _blockwise_cross_corr_flags(
            feature_df=feature_df,
            loaded_features=loaded_features,
            new_features=new_features,
            drop_threshold=threshold,
            report_threshold=0.75,
            block_size=block_size,
            dtype=np.float32,
        )

        if report_pairs:
            print("\n[check_corr] 发现高相关性新特征 (|corr| > 0.75):")
            for new_feat, pairs in report_pairs.items():
                print(f"\n  {new_feat} 与以下特征高度相关:")
                for loaded_feat, corr in pairs:
                    print(f"    - {loaded_feat}: {corr:.3f}")
        else:
            print("\n[check_corr] 没有发现高相关性新特征 (|corr| > 0.75)")

        dropped_features = sorted(dropped_set)
        if drop_flag and dropped_features:
            print(f"\n[check_corr] 删除 {len(dropped_features)} 个高相关性新特征 (|corr| > {threshold}):")
            for feat in dropped_features:
                print(f"  - {feat}")
            feature_df = feature_df.drop(columns=dropped_features)
        elif drop_flag:
            print(f"\n[check_corr] 没有特征超过阈值 |corr| > {threshold}，无需删除")

        return feature_df, dropped_features

    # 旧路径：即时构建小矩阵
    print("[check_corr] 重新计算相关性矩阵（仅使用所需列）")
    all_features = list(loaded_features) + new_features
    corr_matrix_small = feature_df[all_features].corr()
    cross_corr = corr_matrix_small.loc[new_features, loaded_features]

    high_corr_features = cross_corr[(cross_corr.abs() > 0.75).any(axis=1)]
    if not high_corr_features.empty:
        print("\n[check_corr] 发现高相关性新特征 (|corr| > 0.75):")
        for new_feat in high_corr_features.index:
            mask = high_corr_features.loc[new_feat].abs() > 0.75
            correlated_with = high_corr_features.columns[mask]
            corr_values = high_corr_features.loc[new_feat, mask]
            print(f"\n  {new_feat} 与以下特征高度相关:")
            for loaded_feat, corr in zip(correlated_with, corr_values):
                print(f"    - {loaded_feat}: {corr:.3f}")
    else:
        print("\n[check_corr] 没有发现高相关性新特征 (|corr| > 0.75)")

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


def corr_filter(feature_file: str = None,
                threshold: float = 0.95,
                drop_flag: bool = True,
                use_blockwise: bool = True,
                block_size: int = 512,
                intra_group: bool = False,
                intra_threshold: float | None = None):
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
    
    # 4. 相关性计算策略
    if use_blockwise:
        print(f"\n[filter_corr] 启用分块交叉相关（避免构建全量相关矩阵），block_size={block_size}")
    else:
        print(f"\n[filter_corr] 未启用分块，若无预计算矩阵将按需构建小相关矩阵")
    
    # 5. 逐类进行相关性检查（可选组内去相关）
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
        
        # 组内去相关（若启用）
        if intra_group:
            th_intra = threshold if intra_threshold is None else intra_threshold
            kept_in_group, dropped_intra = _intra_group_decorrelate(
                feature_df=feature_df,
                group_features=current_group_features,
                threshold=th_intra,
                report_threshold=0.75,
            )
            if dropped_intra:
                print(f"[filter_corr][intra] 组 {mode_name}_{func_id} 组内去相关删除 {len(dropped_intra)} 列 (|corr| > {th_intra})")
                for feat in dropped_intra:
                    print(f"  - {feat}")
            current_group_features = kept_in_group
            all_dropped_features.extend(dropped_intra)

        if i == 0:
            # 第一类特征仅在启用组内去相关时做组内处理，跳过跨组检查
            print("第一类特征，跳过跨组相关性检查")
            processed_features.extend(current_group_features)
        else:
            # 检查当前组特征与已处理特征的相关性（默认分块计算）
            temp_df, dropped_in_group = check_new_features_corr(
                feature_df,
                processed_features,
                drop_flag=drop_flag,
                threshold=threshold,
                corr_matrix=None,
                use_blockwise=use_blockwise,
                block_size=block_size,
                new_features_list=current_group_features,
            )
            
            all_dropped_features.extend(dropped_in_group)
            
            # 更新已处理特征列表（添加未被删除的当前组特征）
            remaining_group_features = [f for f in current_group_features if f not in dropped_in_group]
            processed_features.extend(remaining_group_features)
            
            print(f"当前组保留特征: {len(remaining_group_features)}")
            print(f"当前组删除特征: {len(dropped_in_group)}")
    
    # 6. 应用删除操作（如果启用）
    final_feature_df = feature_df.copy()
    if drop_flag and all_dropped_features:
        final_feature_df = feature_df.drop(columns=all_dropped_features)
        print(f"\n[filter_corr] 总共删除 {len(all_dropped_features)} 个高相关性特征")
    
    keep_features = [f for f in feature_df.columns if f not in all_dropped_features]
    
    # 7. 保存结果到txt文件
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
        top_k = [5, 10, 15, 20, 100, 200]
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
        top_k = [200, 300, 400, 500, 600, 800]

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