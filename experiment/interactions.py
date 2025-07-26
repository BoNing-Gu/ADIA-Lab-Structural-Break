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
    create_sqmul: bool = False,
    create_add: bool = False,
    create_sub: bool = False,
    create_div: bool = False,
    create_sq: bool = False
):
    """
    根据特征重要性文件生成交互特征。
    支持字典格式的特征数据。

    Args:
        importance_file_path (str): 特征重要性文件路径 (e.g., permutation_importance.tsv)。
        base_feature_file (str, optional): 基础特征文件名。如果提供，
            将加载此文件并在此基础上添加或更新特征。否则，将使用最新的特征集。
        create_mul (bool): 是否创建乘法交互项。默认为 True。
        create_sqmul (bool): 是否创建乘法平方交互项。默认为 False。
        create_add (bool): 是否创建加法交互项。默认为 False。
        create_sub (bool): 是否创建减法交互项。默认为 False。
        create_div (bool): 是否创建除法交互项。默认为 False。
        create_sq (bool): 是否创建平方交互项。默认为 False。
    """
    # 1. 加载重要性文件并获取 Top N 特征
    if len(config.TOP_FEATURES) > 0:
        top_features = config.TOP_FEATURES
        logger.info(f"从配置文件中识别出 Top {len(top_features)} 特征: {top_features}")
    else:
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

    # 2. 确定并加载基础特征文件
    if base_feature_file:
        base_path = config.FEATURE_DIR / base_feature_file
    else:
        base_path = features._get_latest_feature_file()

    if not base_path or not base_path.exists():
        logger.error(f"无法找到基础特征文件。请提供一个有效的文件或确保存在特征文件。")
        return

    logger.info(f"将基于特征文件进行更新: {base_path.name}")
    
    # 尝试加载字典格式的特征文件
    try:
        feature_dict, metadata = features._load_feature_dict_file(base_path)
        is_dict_format = True
        logger.info(f"加载字典格式特征文件，包含数据ID: {list(feature_dict.keys())}")
    except Exception:
        # 回退到旧格式
        feature_df, metadata = features._load_feature_file(base_path)
        if feature_df.empty:
            logger.error("加载的基础特征文件为空，操作中止。")
            return
        feature_dict = {"0": feature_df}
        is_dict_format = False
        logger.info("加载旧格式特征文件，转换为字典格式处理")
        
    initial_feature_counts = {data_id: len(df.columns) for data_id, df in feature_dict.items()}

    # 3. 为每个数据ID生成交互特征
    updated_feature_dict = {}
    all_interaction_features = []
    epsilon = 1e-6
    
    for data_id, feature_df in feature_dict.items():
        logger.info(f"\n为数据ID '{data_id}' 生成交互特征...")
        
        # 检查top features是否都在当前feature_df中
        missing_features = [f for f in top_features if f not in feature_df.columns]
        if missing_features:
            logger.warning(f"数据ID '{data_id}' 中以下 Top 特征缺失，将跳过: {missing_features}")
            available_features = [f for f in top_features if f in feature_df.columns]
            if len(available_features) < 2:
                logger.warning(f"数据ID '{data_id}' 可用特征不足2个，跳过交互特征生成")
                updated_feature_dict[data_id] = feature_df.copy()
                continue
        else:
            available_features = top_features
        
        # 创建交互特征
        interaction_features = pd.DataFrame(index=feature_df.index)
        
        for f1, f2 in combinations(available_features, 2):
            if create_mul:
                interaction_features[f'{f1}_mul_{f2}'] = feature_df[f1] * feature_df[f2]
            if create_sqmul:
                interaction_features[f'{f1}_sqmul_{f2}'] = feature_df[f1] * (feature_df[f2] ** 2)
                interaction_features[f'{f2}_sqmul_{f1}'] = feature_df[f2] * (feature_df[f1] ** 2)
            if create_sub:
                interaction_features[f'{f1}_sub_{f2}'] = feature_df[f1] - feature_df[f2]
            if create_add:
                interaction_features[f'{f1}_add_{f2}'] = feature_df[f1] + feature_df[f2]
            if create_div:
                interaction_features[f'{f1}_div_{f2}'] = feature_df[f1] / (feature_df[f2] + epsilon)
                interaction_features[f'{f2}_div_{f1}'] = feature_df[f2] / (feature_df[f1] + epsilon)

        for f in available_features:
            if create_sq:
                interaction_features[f'{f}_sq'] = feature_df[f] ** 2
        
        if interaction_features.empty:
            logger.info(f"数据ID '{data_id}' 没有选择任何交互项类型，跳过。")
            updated_feature_dict[data_id] = feature_df.copy()
            continue
        
        logger.info(f"  数据ID '{data_id}' 成功创建 {len(interaction_features.columns)} 个交互特征")
        
        # 合并特征
        updated_feature_df = feature_df.drop(columns=interaction_features.columns, errors='ignore')
        updated_feature_df = updated_feature_df.merge(interaction_features, left_index=True, right_index=True, how='left')
        updated_feature_df = features.clean_feature_names(updated_feature_df, prefix="f_inter")
        
        updated_feature_dict[data_id] = updated_feature_df
        
        # 记录所有交互特征名称（用于元数据）
        if data_id == list(feature_dict.keys())[0]:  # 只记录第一个数据ID的交互特征名称
            all_interaction_features = interaction_features.columns.tolist()

    # 4. 检查是否有新特征生成并保存结果
    has_new_features = False
    final_feature_counts = {}
    
    for data_id, df in updated_feature_dict.items():
        final_feature_counts[data_id] = len(df.columns)
        if final_feature_counts[data_id] > initial_feature_counts[data_id]:
            has_new_features = True
    
    if has_new_features:
        if base_path and base_path.exists():
            features._backup_feature_file(base_path)
        
        metadata['last_updated_interaction_file'] = importance_file_path
        metadata['last_interaction_features'] = all_interaction_features
        metadata['data_ids'] = list(updated_feature_dict.keys())
        
        # 保存结果
        if is_dict_format or len(updated_feature_dict) > 1:
            new_file_path = features._save_feature_dict_file(updated_feature_dict, metadata)
        else:
            # 如果原来是单个DataFrame格式且只有一个数据ID，保持兼容性
            new_file_path = features._save_feature_file(updated_feature_dict["0"], metadata)
        
        logger.info(f"交互特征已保存到新文件: {new_file_path.name}")
        
        logger.info("\n=== 交互特征生成完成统计 ===")
        for data_id in updated_feature_dict.keys():
            initial_count = initial_feature_counts[data_id]
            final_count = final_feature_counts[data_id]
            logger.info(f"数据ID '{data_id}': {initial_count} -> {final_count} 个特征 (+{final_count - initial_count})")
        
        # 打印新增的交互特征名列表
        if all_interaction_features:
            logger.info(f"\n新增的交互特征列表 (共 {len(all_interaction_features)} 个):")
            logger.info(f"{all_interaction_features}")
        else:
            logger.info("\n未生成新的交互特征")
    else:
        logger.info("没有新特征生成，文件未保存。")
        new_file_path = None
        


def generate_one2all_interactions(
    base_feature_file: str = None,
    target_feature: str = 'RAW_1_stats_cv_whole'
):
    """
    让指定特征与 config.REMAIN_FEATURES 中所有不含 'mul' 字样的特征进行乘法交互。
    
    Args:
        base_feature_file (str, optional): 基础特征文件名。如果提供，
            将加载此文件并在此基础上添加或更新特征。否则，将使用最新的特征集。
        target_feature (str): 要进行交互的目标特征名，默认为 'RAW_1_stats_cv_whole'。
    """
    # 1. 获取 REMAIN_FEATURES 中不含 'mul'/'sub'/'div'/'add'/'sq' 的特征
    raw_features = [f for f in config.REMAIN_FEATURES if 'mul' not in f and 'sub' not in f and 'div' not in f and 'add' not in f and 'sq' not in f and f != target_feature]
    
    if not raw_features:
        logger.warning(f"在 REMAIN_FEATURES 中没有找到不含 'mul'/'sub'/'div'/'add'/'sq' 且不等于 '{target_feature}' 的特征")
        return
    
    logger.info(f"找到 {len(raw_features)} 个不含 'mul'/'sub'/'div'/'add'/'sq' 的特征将与 '{target_feature}' 进行交互")
    logger.info(f"目标特征: {target_feature}")
    logger.info(f"交互特征列表: {raw_features}")

    # 2. 确定并加载基础特征文件
    if base_feature_file:
        base_path = config.FEATURE_DIR / base_feature_file
    else:
        base_path = features._get_latest_feature_file()

    if not base_path or not base_path.exists():
        logger.error(f"无法找到基础特征文件。请提供一个有效的文件或确保存在特征文件。")
        return

    logger.info(f"将基于特征文件进行更新: {base_path.name}")
    
    # 尝试加载字典格式的特征文件
    try:
        feature_dict, metadata = features._load_feature_dict_file(base_path)
        is_dict_format = True
        logger.info(f"加载字典格式特征文件，包含数据ID: {list(feature_dict.keys())}")
    except Exception:
        # 回退到旧格式
        feature_df, metadata = features._load_feature_file(base_path)
        if feature_df.empty:
            logger.error("加载的基础特征文件为空，操作中止。")
            return
        feature_dict = {"0": feature_df}
        is_dict_format = False
        logger.info("加载旧格式特征文件，转换为字典格式处理")
        
    initial_feature_counts = {data_id: len(df.columns) for data_id, df in feature_dict.items()}

    # 3. 为每个数据ID生成交互特征
    updated_feature_dict = {}
    all_interaction_features = []
    
    for data_id, feature_df in feature_dict.items():
        logger.info(f"\n为数据ID '{data_id}' 生成 {target_feature} 的交互特征...")
        
        # 检查目标特征是否存在
        if target_feature not in feature_df.columns:
            logger.warning(f"数据ID '{data_id}' 中缺少目标特征 '{target_feature}'，跳过")
            updated_feature_dict[data_id] = feature_df.copy()
            continue
        
        # 检查有多少个交互特征在当前feature_df中
        available_features = [f for f in raw_features if f in feature_df.columns]
        missing_features = [f for f in raw_features if f not in feature_df.columns]
        
        if missing_features:
            logger.warning(f"数据ID '{data_id}' 中以下特征缺失，将跳过: {missing_features}")
        
        if not available_features:
            logger.warning(f"数据ID '{data_id}' 中没有可用的交互特征，跳过")
            updated_feature_dict[data_id] = feature_df.copy()
            continue
        
        logger.info(f"  数据ID '{data_id}' 中找到 {len(available_features)} 个可用特征进行交互")
        
        # 创建交互特征
        interaction_features = pd.DataFrame(index=feature_df.index)
        
        for interact_feature in available_features:
            interaction_name = f'{target_feature}_mul_{interact_feature}'
            if interaction_name in feature_df.columns:
                continue
            interaction_features[interaction_name] = feature_df[target_feature] * feature_df[interact_feature]
        
        logger.info(f"  数据ID '{data_id}' 成功创建 {len(interaction_features.columns)} 个交互特征")
        
        # 合并特征（先删除可能存在的同名特征，避免重复）
        updated_feature_df = feature_df.drop(columns=interaction_features.columns, errors='ignore')
        updated_feature_df = updated_feature_df.merge(interaction_features, left_index=True, right_index=True, how='left')
        updated_feature_df = features.clean_feature_names(updated_feature_df, prefix="f_raw1_inter")
        
        updated_feature_dict[data_id] = updated_feature_df
        
        # 记录所有交互特征名称（用于元数据）
        if data_id == list(feature_dict.keys())[0]:  # 只记录第一个数据ID的交互特征名称
            all_interaction_features = interaction_features.columns.tolist()

    # 4. 检查是否有新特征生成并保存结果
    has_new_features = False
    final_feature_counts = {}
    
    for data_id, df in updated_feature_dict.items():
        final_feature_counts[data_id] = len(df.columns)
        if final_feature_counts[data_id] > initial_feature_counts[data_id]:
            has_new_features = True
    
    if has_new_features:
        if base_path and base_path.exists():
            features._backup_feature_file(base_path)
        
        metadata['last_updated_raw1_interaction'] = target_feature
        metadata['last_raw1_interaction_features'] = all_interaction_features
        metadata['data_ids'] = list(updated_feature_dict.keys())
        
        # 保存结果
        if is_dict_format or len(updated_feature_dict) > 1:
            new_file_path = features._save_feature_dict_file(updated_feature_dict, metadata)
        else:
            # 如果原来是单个DataFrame格式且只有一个数据ID，保持兼容性
            new_file_path = features._save_feature_file(updated_feature_dict["0"], metadata)
        
        logger.info(f"RAW1交互特征已保存到新文件: {new_file_path.name}")
        
        logger.info(f"\n=== {target_feature} 交互特征生成完成统计 ===")
        for data_id in updated_feature_dict.keys():
            initial_count = initial_feature_counts[data_id]
            final_count = final_feature_counts[data_id]
            logger.info(f"数据ID '{data_id}': {initial_count} -> {final_count} 个特征 (+{final_count - initial_count})")
        
        # 打印新增的交互特征名列表
        if all_interaction_features:
            logger.info(all_interaction_features)
        else:
            logger.info(f"\n未生成新的 {target_feature} 交互特征")
    else:
        logger.info("没有新特征生成，文件未保存。")
        new_file_path = None
