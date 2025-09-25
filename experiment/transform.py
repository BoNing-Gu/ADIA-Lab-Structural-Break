import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

from . import config, features

try:
    from sklearn.preprocessing import PowerTransformer
except Exception as e:
    PowerTransformer = None

# logger 将由 main.py 在运行时注入
logger = None


def _get_base_feature_path(feature_file: str | None):
    if feature_file:
        return config.FEATURE_DIR / feature_file
    return features._get_latest_feature_file()


def _collect_candidates(feature_dict: Dict[str, pd.DataFrame], candidates: List[str]) -> List[str]:
    # 取所有 data_id 上存在的列的并集与候选集合的交集
    all_cols = set()
    for df in feature_dict.values():
        all_cols.update(df.columns.tolist())
    final_candidates = [c for c in candidates if c in all_cols]
    missing = sorted(set(candidates) - set(final_candidates))
    if missing:
        logger.info(f"REMAIN_FEATURES 中有 {len(missing)} 个列在特征文件中缺失，已跳过: {missing[:20]}{'...' if len(missing) > 20 else ''}")
    return final_candidates


def _compute_global_skew(feature_dict: Dict[str, pd.DataFrame], col: str) -> float:
    series_list = []
    for df in feature_dict.values():
        if col in df.columns:
            series_list.append(df[col])
    if not series_list:
        return 0.0
    concat_vals = pd.concat(series_list, axis=0)
    # 仅对有限数值计算偏度
    vals = pd.to_numeric(concat_vals, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return 0.0
    return float(vals.skew())


def transform_skewed_features(
    feature_file: str | None = None,
    features_to_consider: List[str] | None = None,
    skew_threshold: float | None = None,
    standardize: bool | None = None,
    suffix: str | None = None,
) -> Tuple[str | None, Dict[str, str], List[str], str | None]:
    """
    对 REMAIN_FEATURES 中偏度绝对值超过阈值的特征应用 Yeo-Johnson 变换，
    为每个被转换的特征新增一列，列名添加后缀（默认 _yj），并保存为新的特征文件。

    Returns:
        (new_file_name, mapping_old_to_new, new_remain_features)
    """
    if PowerTransformer is None:
        raise RuntimeError("缺少依赖: scikit-learn。请安装 scikit-learn 以使用 Yeo-Johnson 变换。")

    if features_to_consider is None:
        features_to_consider = config.REMAIN_FEATURES
    if skew_threshold is None:
        skew_threshold = getattr(config, 'SKEWNESS_THRESHOLD', 1.0)
    if standardize is None:
        standardize = getattr(config, 'YJ_STANDARDIZE', True)
    if suffix is None:
        suffix = getattr(config, 'YJ_SUFFIX', '_yj')

    base_path = _get_base_feature_path(feature_file)
    if not base_path or not base_path.exists():
        logger.error("无法找到基础特征文件，请先生成特征或传入 --feature-file。")
        return None, {}, features_to_consider, None

    logger.info(f"将基于特征文件进行Yeo-Johnson更新: {base_path.name}")

    # 加载字典格式（优先）
    try:
        feature_dict, metadata = features._load_feature_dict_file(base_path)
        is_dict_format = True
        logger.info(f"加载字典格式特征文件，包含数据ID: {list(feature_dict.keys())}")
    except Exception:
        feature_df, metadata = features._load_feature_file(base_path)
        if feature_df.empty:
            logger.error("加载的基础特征文件为空，操作中止。")
            return None, {}, features_to_consider, None
        feature_dict = {"0": feature_df}
        is_dict_format = False
        logger.info("加载旧格式特征文件，转换为字典格式处理")

    # 候选列
    candidates = _collect_candidates(feature_dict, features_to_consider)
    # 输出总特征数量与 REMAIN_FEATURES 数量
    all_cols_union = set()
    for df in feature_dict.values():
        all_cols_union.update(df.columns.tolist())
    total_feature_count = len(all_cols_union)
    logger.info(f"当前特征总数: {total_feature_count}")
    logger.info(f"REMAIN_FEATURES 数量: {len(features_to_consider)}")
    print(f"Total features: {total_feature_count}")
    print(f"REMAIN_FEATURES count: {len(features_to_consider)}")
    if not candidates:
        logger.warning("没有可用的候选列，操作中止。")
        return None, {}, features_to_consider, None

    # 计算全局偏度，并筛选需要转换的列
    skew_map: Dict[str, float] = {}
    to_transform: List[str] = []
    for col in candidates:
        s = _compute_global_skew(feature_dict, col)
        skew_map[col] = s
        if np.isfinite(s) and abs(s) > float(skew_threshold):
            to_transform.append(col)

    if not to_transform:
        logger.info(f"没有发现偏度绝对值大于 {skew_threshold} 的特征，未进行任何转换。")
        return None, {}, features_to_consider, None

    logger.info(f"偏度 |skew|>{skew_threshold} 的特征数量: {len(to_transform)}")
    logger.info(f"示例: {to_transform[:30]}{'...' if len(to_transform) > 30 else ''}")

    # 逐列拟合全局变换器，并在每个 data_id 上生成新列
    mapping_old_to_new: Dict[str, str] = {}
    updated_feature_dict: Dict[str, pd.DataFrame] = {k: v.copy() for k, v in feature_dict.items()}

    for col in to_transform:
        # 全局拟合
        concat_vals = []
        for df in feature_dict.values():
            if col in df.columns:
                concat_vals.append(pd.to_numeric(df[col], errors='coerce'))
        concat_series = pd.concat(concat_vals, axis=0).replace([np.inf, -np.inf], np.nan).dropna()
        if concat_series.empty:
            logger.info(f"列 {col} 有效数值为空，跳过。")
            continue

        # 常量列跳过
        if float(np.nanstd(concat_series.values)) < 1e-12:
            logger.info(f"列 {col} 近似常量（std≈0），跳过。")
            continue

        try:
            pt = PowerTransformer(method='yeo-johnson', standardize=bool(standardize))
            pt.fit(concat_series.values.reshape(-1, 1))
        except Exception:
            logger.info(f"列 {col} 拟合 Yeo-Johnson 失败，跳过。")
            continue

        new_col = f"{col}{suffix}"
        mapping_old_to_new[col] = new_col

        # 对每个 data_id 进行转换并写入新列
        for data_id, df in updated_feature_dict.items():
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
            mask = vals.notna()
            transformed = np.full(shape=vals.shape, fill_value=np.nan, dtype=float)
            if mask.any():
                try:
                    transformed_vals = pt.transform(vals[mask].values.reshape(-1, 1)).ravel()
                    transformed[mask.to_numpy()] = transformed_vals
                except Exception:
                    logger.info(f"列 {col} 在数据ID {data_id} 转换失败，保留 NaN。")
            # 写入新列（如存在则覆盖）
            df[new_col] = transformed

    if not mapping_old_to_new:
        logger.info("没有任何列成功转换，操作结束。")
        return None, {}, features_to_consider, None

    # 保存
    if base_path and base_path.exists():
        features._backup_feature_file(base_path)

    metadata = metadata or {}
    metadata['last_yj_transformed_columns'] = mapping_old_to_new
    metadata['skew_threshold'] = float(skew_threshold)
    metadata['yj_standardize'] = bool(standardize)
    metadata['yj_suffix'] = suffix

    if is_dict_format or len(updated_feature_dict) > 1:
        new_path = features._save_feature_dict_file(updated_feature_dict, metadata)
    else:
        new_path = features._save_feature_file(updated_feature_dict["0"], metadata)

    logger.info(f"已保存应用 Yeo-Johnson 转换的新特征文件: {new_path.name}")
    logger.info("原始列已保留；新增列使用后缀。")

    # 生成新的 REMAIN_FEATURES 列表：优先使用转换后的列名
    new_remain = []
    for col in features_to_consider:
        new_remain.append(mapping_old_to_new.get(col, col))

    # 在 output 目录记录转换后应当使用的特征列表
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = config.OUTPUT_DIR / f"transform_features_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / f"remain_features_{ts}.txt"
    try:
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write(f"# Generated at: {ts}\n")
            f.write(f"# Base feature file: {base_path.name}\n")
            f.write(f"# New feature file: {new_path.name}\n")
            f.write(f"# Total features: {total_feature_count}\n")
            f.write(f"# REMAIN_FEATURES count: {len(features_to_consider)}\n")
            f.write(f"# Transformed columns: {len(mapping_old_to_new)}\n")
            f.write("\n")
            f.write("REMAIN_FEATURES = [\n")
            for col in new_remain:
                f.write("    '" + str(col).replace("'", "\\'") + "',\n")
            f.write("]\n")
        logger.info(f"转换后的特征列表已写入: {out_txt}")
        out_txt_path = str(out_txt)
    except Exception:
        out_txt_path = None

    return new_path.name, mapping_old_to_new, new_remain, out_txt_path


