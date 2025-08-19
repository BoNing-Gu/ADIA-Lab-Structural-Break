import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from . import config, utils

logger = None

def load_data(enhancement_ids=None, return_dict=True):
    """
    加载训练数据，根据 return_dict 参数返回不同格式的数据
    
    Args:
        enhancement_ids: list, 可选的增强数据ID列表，如果提供则会加载对应的增强数据并合并
        return_dict: bool, 是否返回字典格式。True返回字典，False返回拼接后的单个DataFrame和Series
    
    Returns:
        tuple: (X_data, y_data) 其中：
            - 当 return_dict=True 时：
                - X_data: dict, 以数据ID为键的DataFrame字典
                - y_data: dict, 以数据ID为键的Series字典
            - 当 return_dict=False 时：
                - X_data: DataFrame, 拼接后的特征数据
                - y_data: Series, 拼接后的标签数据
    """
    try:
        logger.info("Loading data...")
        X_train = pd.read_parquet(config.TRAIN_X_FILE)
        y_train = pd.read_parquet(config.TRAIN_Y_FILE)
        logger.info("Original data loaded successfully.")
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_train type: {type(X_train)}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"y_train type: {type(y_train)}")
        
        # 初始化字典格式的数据
        X_data = {}
        y_data = {}
        
        # 如果指定了增强数据ID，则加载并合并增强数据
        if enhancement_ids:
            logger.info(f"Loading enhancement data with IDs: {enhancement_ids}")
            for func_id in enhancement_ids:
                if func_id == "0":
                    X_data[str(func_id)] = X_train
                    y_data[str(func_id)] = y_train
                else:
                    enhance_x_file = config.DATA_DIR / f'X_train_enhance{func_id}.parquet'
                    enhance_y_file = config.DATA_DIR / f'y_train_enhance{func_id}.parquet'
                    
                    if enhance_x_file.exists() and enhance_y_file.exists():
                        X_enhance = pd.read_parquet(enhance_x_file)
                        y_enhance_df = pd.read_parquet(enhance_y_file)
                        
                        # 确保只取第一列并转换为Series，保持原始索引
                        if isinstance(y_enhance_df, pd.DataFrame):
                            if 'structural_breakpoint' in y_enhance_df.columns:
                                y_enhance = y_enhance_df['structural_breakpoint']
                            else:
                                y_enhance = y_enhance_df.iloc[:, 0]
                        else:
                            y_enhance = y_enhance_df
                        
                        # 确保y_enhance是Series类型
                        if not isinstance(y_enhance, pd.Series):
                            y_enhance = pd.Series(y_enhance, name='structural_breakpoint')
                        
                        # 添加到字典中
                        X_data[str(func_id)] = X_enhance
                        y_data[str(func_id)] = y_enhance
                        
                        logger.info(f"Enhancement data {func_id} loaded and added.")
                        logger.info(f"New X_enhance shape: {X_enhance.shape}")
                        logger.info(f"New y_enhance shape: {y_enhance.shape}")
                    else:
                        logger.warning(f"Enhancement data files for {func_id} not found, skipping.")
        else:
            # 如果没有指定增强数据ID，只加载原始数据
            X_data["0"] = X_train
            y_data["0"] = y_train
        
        logger.info(f"Total datasets loaded: {len(X_data)}")
        for data_id in X_data.keys():
            logger.info(f"Dataset {data_id}: X shape {X_data[data_id].shape}, y shape {y_data[data_id].shape}")
    
        if return_dict:
            return X_data, y_data
        else:
            # 合并数据时保持原始索引，不重置索引
            X_combined = pd.concat(list(X_data.values()), axis=0, ignore_index=False)
            y_combined = pd.concat(list(y_data.values()), axis=0, ignore_index=False)
            
            logger.info(f"Return CONCATED data shape: X {X_combined.shape}, y {y_combined.shape}")
            return X_combined, y_combined
            
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise 

# --- 数据增强函数注册表 ---
DATA_ENHANCEMENT_REGISTRY = {}

def register_data_enhancement(_func=None, *, times=1, func_id=""):
    """一个用于注册数据增强函数的装饰器，可以标记方法的增强次数。"""
    def decorator_register(func):
        DATA_ENHANCEMENT_REGISTRY[func.__name__] = {
            "func": func, 
            "times": times,
            "func_id": func_id
        }
        return func

    if _func is None:
        # Used as @register_data_enhancement(times=..., func_id=...)
        return decorator_register
    else:
        # Used as @register_data_enhancement
        return decorator_register(_func)

@register_data_enhancement(times=1, func_id="1")
def scale_augmentation(u: pd.DataFrame, y: bool, id: int) -> dict:
    """
    基础的缩放数据增强函数
    
    Args:
        u: 单个id对应的DataFrame，包含time, value, period列
        y: 单个id对应的标签，bool类型
    
    Returns:
        dict: 包含增强后的数据，格式为 {'X_enhanced': DataFrame, 'y_enhanced': Series}
    """
    enhanced_data = []
    enhanced_labels = []
    
    # 获取注册信息
    func_info = DATA_ENHANCEMENT_REGISTRY['scale_augmentation']
    times = func_info['times']
    func_id = func_info['func_id']
    
    for i in range(times):
        # 从(0.5,1.5)∪(-1.5,-0.5)之间取出一个随机数
        if np.random.random() < 0.5:
            scale_factor = np.random.uniform(0.5, 1.5)
        else:
            scale_factor = np.random.uniform(-1.5, -0.5)
        
        # 复制原始数据
        u_aug = u.copy()
        
        if y:  # 如果有结构断点
            # 随机选择period=0段或period=1段
            if np.random.random() < 0.5:
                # 选择period=0段
                mask = u_aug['period'] == 0
            else:
                # 选择period=1段
                mask = u_aug['period'] == 1
            u_aug.loc[mask, 'value'] *= scale_factor
        else:  # 如果没有结构断点
            # 对整段时序value乘以该数
            u_aug['value'] *= scale_factor
        
        # 生成新的ID，确保与原始数据类型一致（int64）
        new_id = int(func_id) * 1000000 + i * 100000 + int(id)
        
        # 重置索引并设置新的ID
        if isinstance(u_aug.index, pd.MultiIndex):
            u_aug = u_aug.reset_index()
            u_aug['id'] = new_id
            # 确保数据类型一致
            u_aug['id'] = u_aug['id'].astype('int64')
            u_aug['time'] = u_aug['time'].astype('int64')
            u_aug['value'] = u_aug['value'].astype('float64')
            u_aug['period'] = u_aug['period'].astype('int64')
            u_aug = u_aug.set_index(['id', 'time'])
        else:
            u_aug = u_aug.reset_index()
            u_aug['id'] = new_id
            # 确保数据类型一致
            u_aug['id'] = u_aug['id'].astype('int64')
            u_aug['time'] = u_aug['time'].astype('int64')
            u_aug['value'] = u_aug['value'].astype('float64')
            u_aug['period'] = u_aug['period'].astype('int64')
            u_aug = u_aug.set_index(['id', 'time'])
        
        enhanced_data.append(u_aug)
        # 创建标签Series，确保索引类型为int64
        enhanced_labels.append(pd.Series([y], index=pd.Index([new_id], dtype='int64'), name='structural_breakpoint'))
    
    # 合并所有增强数据
    u_enhanced = pd.concat(enhanced_data, axis=0)
    y_enhanced = pd.concat(enhanced_labels, axis=0)
    
    return {'X_enhanced': u_enhanced, 'y_enhanced': y_enhanced}

@register_data_enhancement(times=1, func_id="2")
def noise_augmentation(u: pd.DataFrame, y: bool, id: int) -> dict:
    """
    基于时序数据特征的自适应噪声增强函数
    
    Args:
        u: 单个id对应的DataFrame，包含time, value, period列
        y: 单个id对应的标签，bool类型
    
    Returns:
        dict: 包含增强后的数据，格式为 {'X_enhanced': DataFrame, 'y_enhanced': Series}
    """
    enhanced_data = []
    enhanced_labels = []
    
    # 获取注册信息
    func_info = DATA_ENHANCEMENT_REGISTRY['noise_augmentation']
    times = func_info['times']
    func_id = func_info['func_id']
    
    for i in range(times):
        # 复制原始数据
        u_aug = u.copy()
        
        # 计算时序数据的统计特征来确定噪声强度
        value_std = u_aug['value'].std()
        value_mean = abs(u_aug['value'].mean())
        value_range = u_aug['value'].max() - u_aug['value'].min()
        
        # 自适应噪声强度计算
        # 基础噪声强度为标准差的一定比例
        base_noise_ratio = np.random.uniform(0.05, 0.15)  # 5%-15%的标准差
        noise_std = value_std * base_noise_ratio
        
        # 如果标准差太小，使用值域作为参考
        if noise_std < 1e-6:
            noise_std = value_range * np.random.uniform(0.01, 0.05)  # 1%-5%的值域
        
        # 如果值域也很小，使用均值作为参考
        if noise_std < 1e-6:
            noise_std = abs(value_mean) * np.random.uniform(0.01, 0.03)  # 1%-3%的均值
        
        # 最小噪声保证
        min_noise = 1e-4
        noise_std = max(noise_std, min_noise)
        
        if y:  # 如果有结构断点
            # 对不同period段添加不同强度的噪声
            period_0_mask = u_aug['period'] == 0
            period_1_mask = u_aug['period'] == 1
            
            # 为两个period段生成不同的噪声强度
            noise_ratio_0 = np.random.uniform(0.8, 1.2)  # period 0的噪声倍数
            noise_ratio_1 = np.random.uniform(0.8, 1.2)  # period 1的噪声倍数
            
            # 添加噪声
            if period_0_mask.any():
                noise_0 = np.random.normal(0, noise_std * noise_ratio_0, period_0_mask.sum())
                u_aug.loc[period_0_mask, 'value'] += noise_0
            
            if period_1_mask.any():
                noise_1 = np.random.normal(0, noise_std * noise_ratio_1, period_1_mask.sum())
                u_aug.loc[period_1_mask, 'value'] += noise_1
                
        else:  # 如果没有结构断点
            # 对整个时序添加一致的噪声
            noise_ratio = np.random.uniform(0.8, 1.2)  # 噪声强度的随机变化
            noise = np.random.normal(0, noise_std * noise_ratio, len(u_aug))
            u_aug['value'] += noise
        
        # 可选：添加时间相关的噪声模式
        if np.random.random() < 0.3:  # 30%的概率添加时间相关噪声
            # 创建一个缓慢变化的噪声趋势
            time_trend_noise = np.sin(np.linspace(0, 2*np.pi, len(u_aug))) * noise_std * 0.5
            u_aug['value'] += time_trend_noise
        
        # 生成新的ID，确保与原始数据类型一致（int64）
        new_id = int(func_id) * 1000000 + i * 100000 + int(id)
        
        # 重置索引并设置新的ID
        if isinstance(u_aug.index, pd.MultiIndex):
            u_aug = u_aug.reset_index()
            u_aug['id'] = new_id
            # 确保数据类型一致
            u_aug['id'] = u_aug['id'].astype('int64')
            u_aug['time'] = u_aug['time'].astype('int64')
            u_aug['value'] = u_aug['value'].astype('float64')
            u_aug['period'] = u_aug['period'].astype('int64')
            u_aug = u_aug.set_index(['id', 'time'])
        else:
            u_aug = u_aug.reset_index()
            u_aug['id'] = new_id
            # 确保数据类型一致
            u_aug['id'] = u_aug['id'].astype('int64')
            u_aug['time'] = u_aug['time'].astype('int64')
            u_aug['value'] = u_aug['value'].astype('float64')
            u_aug['period'] = u_aug['period'].astype('int64')
            u_aug = u_aug.set_index(['id', 'time'])
        
        enhanced_data.append(u_aug)
        # 创建标签Series，确保索引类型为int64
        enhanced_labels.append(pd.Series([y], index=pd.Index([new_id], dtype='int64'), name='structural_breakpoint'))
    
    # 合并所有增强数据
    u_enhanced = pd.concat(enhanced_data, axis=0)
    y_enhanced = pd.concat(enhanced_labels, axis=0)
    
    return {'X_enhanced': u_enhanced, 'y_enhanced': y_enhanced}

@register_data_enhancement(times=1, func_id="3")
def swap_periods_augmentation(u: pd.DataFrame, y: bool, id: int) -> dict:
    """
    交换 period 0 和 period 1 的数据段。
    此增强仅在存在结构断点时应用 (y=True)。
    
    Args:
        u: 单个id对应的DataFrame，包含time, value, period列
        y: 单个id对应的标签，bool类型
    
    Returns:
        dict: 包含增强后的数据，格式为 {'X_enhanced': DataFrame, 'y_enhanced': Series}
    """
    # 如果没有结构断点，则不进行增强，直接返回空结果
    if not y:
        return {'X_enhanced': pd.DataFrame(), 'y_enhanced': pd.Series(dtype='bool')}

    func_info = DATA_ENHANCEMENT_REGISTRY['swap_periods_augmentation']
    times = func_info['times']
    func_id = func_info['func_id']

    enhanced_data = []
    enhanced_labels = []

    for i in range(times):
        u_aug = u.copy()
        
        period_0_df = u_aug[u_aug['period'] == 0].copy()
        period_1_df = u_aug[u_aug['period'] == 1].copy()

        # 如果任一段为空，则不进行增强
        if period_0_df.empty or period_1_df.empty:
            continue

        # 新的时间轴和period
        len_1 = len(period_1_df)
        len_0 = len(period_0_df)

        # 将原来的period 1 放到前面
        period_1_df['time'] = range(len_1)
        period_1_df['period'] = 0

        # 将原来的period 0 放到后面
        period_0_df['time'] = range(len_1, len_1 + len_0)
        period_0_df['period'] = 1

        # 合并
        u_swapped = pd.concat([period_1_df, period_0_df])

        # 生成新的ID
        new_id = int(func_id) * 1000000 + i * 100000 + int(id)
        
        # 重置索引并设置新的ID
        if isinstance(u_swapped.index, pd.MultiIndex):
            u_swapped = u_swapped.reset_index(drop=True)
        u_swapped['id'] = new_id
        u_swapped['id'] = u_swapped['id'].astype('int64')
        u_swapped = u_swapped.set_index(['id', 'time'])

        enhanced_data.append(u_swapped)
        enhanced_labels.append(pd.Series([y], index=pd.Index([new_id], dtype='int64'), name='structural_breakpoint'))

    if not enhanced_data:
        return {'X_enhanced': pd.DataFrame(), 'y_enhanced': pd.Series(dtype='bool')}
        
    u_enhanced = pd.concat(enhanced_data, axis=0)
    y_enhanced = pd.concat(enhanced_labels, axis=0)

    return {'X_enhanced': u_enhanced, 'y_enhanced': y_enhanced}

def _apply_data_enhancement_sequential(func, X_df: pd.DataFrame, y_df: pd.DataFrame) -> tuple:
    """
    顺序应用单个数据增强函数
    
    Args:
        func: 数据增强函数
        X_df: 原始X数据
        y_df: 原始y数据
    
    Returns:
        tuple: (增强后的X_df, 增强后的y_df)
    """
    all_ids = X_df.index.get_level_values("id").unique()
    
    all_X_enhanced = []
    all_y_enhanced = []
    
    for id_val in tqdm(all_ids, desc=f"Running {func.__name__} (sequentially)"):
        u = X_df.loc[id_val]
        y = y_df.loc[id_val]
        
        # 如果y_single是Series，取第一个值
        if isinstance(y, pd.Series):
            y = y.iloc[0]
        
        # 应用增强函数
        result = func(u, y, id_val)
        
        # 跳过空的增强结果
        if result['X_enhanced'].empty:
            continue

        all_X_enhanced.append(result['X_enhanced'])
        all_y_enhanced.append(result['y_enhanced'])
    
    # 合并所有结果
    X_enhanced_final = pd.concat(all_X_enhanced, axis=0)
    y_enhanced_final = pd.concat(all_y_enhanced, axis=0)
    
    return X_enhanced_final, y_enhanced_final

def apply_data_enhancement(func_names=None):
    """
    应用数据增强并保存结果
    
    Args:
        func_names: list, 要应用的增强函数名列表，如果为None则应用所有注册的函数
    """
    # 确保data目录存在
    config.DATA_DIR.mkdir(exist_ok=True)
    
    # 加载原始数据
    logger.info("Loading original data for enhancement...")
    X_data, y_data = load_data()
    X_train, y_train = X_data["0"], y_data["0"]
    
    # 确定要运行的函数
    if func_names is None:
        func_names = list(DATA_ENHANCEMENT_REGISTRY.keys())
    
    logger.info(f"Applying data enhancement functions: {func_names}")
    
    for func_name in func_names:
        if func_name not in DATA_ENHANCEMENT_REGISTRY:
            logger.warning(f"Function {func_name} not found in registry, skipping.")
            continue
        
        logger.info(f"Applying enhancement function: {func_name}")
        
        func_info = DATA_ENHANCEMENT_REGISTRY[func_name]
        func = func_info['func']
        func_id = func_info['func_id']
        
        # 应用增强函数
        X_enhanced, y_enhanced = _apply_data_enhancement_sequential(func, X_train, y_train)
        
        # 保存增强数据
        enhance_x_file = config.DATA_DIR / f'X_train_enhance{func_id}.parquet'
        enhance_y_file = config.DATA_DIR / f'y_train_enhance{func_id}.parquet'
        
        X_enhanced.to_parquet(enhance_x_file)
        # 将Series转换为DataFrame后保存
        y_enhanced_df = y_enhanced.to_frame()
        y_enhanced_df.to_parquet(enhance_y_file)
        
        logger.info(f"Enhancement data saved:")
        logger.info(f"  X: {enhance_x_file}")
        logger.info(f"  y: {enhance_y_file}")
        logger.info(f"  Enhanced X shape: {X_enhanced.shape}")
        logger.info(f"  Enhanced y shape: {y_enhanced.shape}")
    
    logger.info("Data enhancement completed.")
