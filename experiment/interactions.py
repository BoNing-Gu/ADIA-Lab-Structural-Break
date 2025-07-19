import pandas as pd
from itertools import combinations
from . import config, features
import json

# logger 将由 main.py 在运行时注入
logger = None

def generate_interaction_features(
    importance_file_path: str, 
    base_feature_file: str = None,
    create_mul: bool = True,
    create_add: bool = False,
    create_sub: bool = False,
    create_div: bool = False
):
    """
    根据特征重要性文件生成交互特征。

    Args:
        importance_file_path (str): 特征重要性文件路径 (e.g., permutation_importance.tsv)。
        base_feature_file (str, optional): 基础特征文件名。如果提供，
            将加载此文件并在此基础上添加或更新特征。否则，将使用最新的特征集。
        create_mul (bool): 是否创建乘法交互项。默认为 True。
        create_add (bool): 是否创建加法交互项。默认为 False。
        create_sub (bool): 是否创建减法交互项。默认为 False。
        create_div (bool): 是否创建除法交互项。默认为 False。
    """
    # 1. 加载重要性文件并获取 Top N 特征
    try:
        importance_df = pd.read_csv(importance_file_path, sep='\\t', engine='python')
    except FileNotFoundError:
        logger.error(f"重要性文件未找到: {importance_file_path}")
        return

    # 确定重要性列名，兼容多种格式
    if 'permutation_importance_mean' in importance_df.columns:
        importance_col = 'permutation_importance_mean'
    elif 'importance' in importance_df.columns:
        importance_col = 'importance'
    elif 'gain' in importance_df.columns:
        importance_col = 'gain'
    else:
        logger.error(f"在文件中找不到有效的特征重要性列 (e.g., 'permutation_importance_mean', 'importance', 'gain'): {importance_file_path}")
        return

    top_features = importance_df.sort_values(by=importance_col, ascending=False).head(10)['feature'].tolist()
    logger.info(f"从 {importance_file_path} 中识别出 Top 10 特征: {top_features}")

    # 2. 加载基础特征文件
    feature_df, base_file_name = features.load_features(base_feature_file)
    if feature_df is None:
        logger.error(f"无法加载特征文件，操作中止。")
        return
    base_path = config.FEATURE_DIR / base_file_name
    initial_feature_count = len(feature_df.columns)

    missing_features = [f for f in top_features if f not in feature_df.columns]
    if missing_features:
        logger.error(f"以下 Top 特征在基础特征文件中缺失，无法创建交互项: {missing_features}")
        return

    # 3. 创建交互特征
    interaction_features = pd.DataFrame(index=feature_df.index)
    epsilon = 1e-6

    for f1, f2 in combinations(top_features, 2):
        if create_mul:
            interaction_features[f'{f1}_mul_{f2}'] = feature_df[f1] * feature_df[f2]
        if create_sub:
            interaction_features[f'{f1}_sub_{f2}'] = feature_df[f1] - feature_df[f2]
        if create_add:
            interaction_features[f'{f1}_add_{f2}'] = feature_df[f1] + feature_df[f2]
        if create_div:
            interaction_features[f'{f1}_div_{f2}'] = feature_df[f1] / (feature_df[f2] + epsilon)
            interaction_features[f'{f2}_div_{f1}'] = feature_df[f2] / (feature_df[f1] + epsilon)

    if interaction_features.empty:
        logger.info("没有选择任何交互项类型，操作中止。")
        return

    logger.info(f"成功创建 {len(interaction_features.columns)} 个交互特征。")
    logger.info("--- 新生成的交互特征列表 ---")
    logger.info(interaction_features.columns.tolist())
    logger.info("------------------------------")

    # 4. 合并并保存
    feature_df = feature_df.drop(columns=interaction_features.columns, errors='ignore')
    feature_df = feature_df.merge(interaction_features, left_index=True, right_index=True, how='left')
    feature_df = features.clean_feature_names(feature_df, prefix="f_inter")

    # 5. 保存结果
    new_feature_count = len(feature_df.columns)
    if new_feature_count > initial_feature_count:
        if base_path and base_path.exists():
            features._backup_feature_file(base_path)
        
        metadata = json.loads(feature_df.attrs.get('feature_metadata', '{}'))
        metadata['last_updated_interaction_file'] = importance_file_path
        metadata['last_interaction_features'] = interaction_features.columns.tolist()
        
        new_file_path = features._save_feature_file(feature_df, metadata)
        logger.info(f"交互特征已保存到新文件: {new_file_path.name}")
    else:
        logger.info("没有新特征生成，文件未保存。")
        
    final_feature_count = len(feature_df.columns)
    new_file_name = new_file_path.name if 'new_file_path' in locals() and new_feature_count > initial_feature_count else 'N/A'
    logger.info(f"交互特征生成完成。新文件: {new_file_name}, 总特征数: {final_feature_count}")