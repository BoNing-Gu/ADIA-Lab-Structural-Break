import pandas as pd
import numpy as np
from . import config, features, data
import logging
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from umap import UMAP
from tqdm.auto import tqdm

# logger 将由 main.py 在运行时注入
logger = None

class DimReducer:
    def __init__(self, reducer_method='PCA', reducer_params=None):
        """
        reducer: 降维器
        reducer_params: dict，降维器的初始化参数
        """
        if reducer_method == 'PCA':
            self.reducer_class = PCA
        elif reducer_method == 'UMAP':
            self.reducer_class = UMAP
        self.reducer_params = reducer_params if reducer_params is not None else {'n_components': 2}
        self.model = None
        self.feature_names_ = None  # 记录参与降维的特征名

    def fit(self, X_train):
        # 使用所有特征列进行降维
        self.feature_names_ = list(X_train.columns)
        if len(self.feature_names_) == 0:
            raise ValueError("No features found in input data.")
        self.model = self.reducer_class(**self.reducer_params)
        self.model.fit(X_train.values)

    def predict(self, x_query):
        """
        x_query: 1D array-like，对应的是完整的特征行
        返回降维后的 1D numpy array
        """
        assert self.model is not None, "Model not fitted yet."
        if isinstance(x_query, pd.Series):
            x_query = x_query[self.feature_names_].values.reshape(1, -1)
        else:
            x_query = np.asarray(x_query).reshape(1, -1)
        return self.model.transform(x_query).flatten()

    def extract(self, feature_df):
        """
        对整个 DataFrame 做降维，返回新的 DataFrame 和 新特征名列表
        """
        assert self.model is not None, "Model not fitted yet."
        reduced = self.model.transform(feature_df.values)
        reduced_cols = [f'dim_{i}' for i in range(reduced.shape[1])]
        reduced_df = pd.DataFrame(reduced, columns=reduced_cols, index=feature_df.index)
        feature_df = pd.concat([feature_df, reduced_df], axis=1)
        return feature_df, reduced_cols

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'reducer_class': self.reducer_class,
                'reducer_params': self.reducer_params,
                'model': self.model,
                'feature_names_': self.feature_names_
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        self.reducer_class = obj['reducer_class']
        self.reducer_params = obj['reducer_params']
        self.model = obj['model']
        self.feature_names_ = obj['feature_names_']
    
    def plot(self, reduced_df, oof_df, save_path=None):
        """
        绘制二维降维结果，使用OOF预测值进行颜色标注
        
        Args:
            reduced_df: 降维后的DataFrame，包含dim_0, dim_1等列
            oof_df: OOF预测值DataFrame，包含预测值列
            save_path: 保存图片的路径，如果为None则显示图片
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 检查是否为二维降维
        if self.reducer_params.get('n_components', 2) != 2:
            logger.warning("plot方法仅支持二维降维可视化")
            return
        
        # 检查是否有dim_0和dim_1列
        if 'dim_0' not in reduced_df.columns or 'dim_1' not in reduced_df.columns:
            logger.error("降维结果中缺少dim_0或dim_1列")
            return
        
        # 对齐数据
        common_index = reduced_df.index.intersection(oof_df.index)
        if len(common_index) == 0:
            logger.warning("降维结果与OOF预测值无共同索引，无法绘制")
            return
        
        reduced_aligned = reduced_df.loc[common_index]
        oof_aligned = oof_df.loc[common_index]
        
        # 获取预测值列（假设第一列是预测值）
        pred_col = 'oof_preds'
        predictions = oof_aligned[pred_col].values
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 绘制散点图，使用预测值作为颜色
        scatter = plt.scatter(reduced_aligned['dim_0'], 
                            reduced_aligned['dim_1'], 
                            c=predictions, 
                            cmap='viridis', 
                            alpha=0.7, 
                            s=20)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter)
        cbar.set_label(f'OOF Predictions ({pred_col})', rotation=270, labelpad=20)
        
        # 设置标题和标签
        plt.xlabel('dim_0', fontsize=12)
        plt.ylabel('dim_1', fontsize=12)
        
        # 添加网格
        plt.grid(True, alpha=0.3)
        
        # 设置布局
        plt.tight_layout()
        
        # 保存或显示图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"降维可视化图片已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # 打印统计信息
        logger.info(f"降维可视化完成:")
        logger.info(f"  - 数据点数量: {len(common_index)}")
        logger.info(f"  - 预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
        logger.info(f"  - 预测值均值: {predictions.mean():.4f}")
        logger.info(f"  - 预测值标准差: {predictions.std():.4f}")

class NeighborFeatureExtractor:
    def __init__(self, metric='euclidean', stage='train'):
        self.metric = metric
        self.nn_model = None
        self.X_train = None
        self.stage = stage
        self.add_n_neighbors = 1 if self.stage == 'train' else 0

    def fit(self, X_train, n_neighbors=20):
        """
        X_train: pd.DataFrame
        n_neighbors: 最大邻居数（内部存储）
        """
        self.X_train = X_train.copy()
        self.nn_model = NearestNeighbors(
            n_neighbors=n_neighbors + self.add_n_neighbors,
            metric=self.metric,
            n_jobs=config.N_JOBS
        )
        self.nn_model.fit(self.X_train.values)

    def predict(self, x_query, neighbor_list=[10]):
        """
        x_query: 1D array-like, shape (n_features,)
        neighbor_list: list of ints, 每个值是希望提取的邻居个数
        返回：dict，形如 {'featname_nn_3_mean': xxx, 'featname_nn_5_mean': xxx, ...}
        """
        assert self.nn_model is not None, "Model not fitted yet."
        x_query = np.asarray(x_query).reshape(1, -1)
        distances, indices = self.nn_model.kneighbors(x_query)
        
        if self.stage == 'train':
            # indices[0][0] 是自身，跳过
            neighbor_indices = indices[0][1:]
        else:
            # 不跳过
            neighbor_indices = indices[0][0:]

        feats = {}
        for k in neighbor_list:
            if k > len(neighbor_indices):
                for col in self.X_train.columns:
                    feats[f'{col}_nn_{k}_mean'] = np.nan
            else:
                neighbor_idx = neighbor_indices[:k]
                neighbor_X = self.X_train.iloc[neighbor_idx]
                for col in self.X_train.columns:
                    feats[f'{col}_nn_{k}_mean'] = neighbor_X[col].mean()
        return feats

    def extract(self, feature_df, neighbor_list=[10]):
        feature_df = feature_df.copy()
        tqdm.pandas(desc="提取最近邻特征")
        extracted_feature_df = feature_df.progress_apply(
            lambda row: pd.Series(self.predict(row, neighbor_list)),
            axis=1
        )
        new_features = extracted_feature_df.columns.tolist()

        return extracted_feature_df, new_features

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'metric': self.metric,
                'X_train': self.X_train,
                'nn_model': self.nn_model
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        self.metric = obj['metric']
        self.X_train = obj['X_train']
        self.nn_model = obj['nn_model']

def generate_dim_reduction_neighbor_features(
    base_feature_file: str = None,
    oof_file: str = None,
    reducer_method: str = 'PCA',
    n_components: int = 42,
    n_neighbors: int = 20,
    neighbor_list: list = [10],
    metric: str = 'euclidean',
    stage: str = 'train'
):
    """
    基于降维和最近邻方法生成新特征。
    支持字典格式的特征数据。

    Args:
        base_feature_file (str, optional): 基础特征文件名。如果提供，
            将加载此文件并在此基础上添加或更新特征。否则，将使用最新的特征集。
        oof_file (str, optional): OOF预测文件名。如果提供，将用于可视化展示。

        n_components (int): 降维后的组件数量。默认为 42。
        n_neighbors (int): 最近邻模型的最大邻居数。默认为 50。
        neighbor_list (list): 要提取的邻居个数列表。默认为 [3, 5, 7, 10]。
        metric (str): 最近邻距离度量。默认为 'euclidean'。
        stage (str): 阶段标识，'train' 或 'test'。默认为 'train'。
    """
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

    # 3. 为每个数据ID生成降维和最近邻特征
    updated_feature_dict = {}
    all_new_features = []
    
    for data_id, feature_df in feature_dict.items():
        logger.info(f"\n为数据ID '{data_id}' 生成降维和最近邻特征...")
        
        logger.info(f"使用所有 {len(config.REMAIN_FEATURES)} 个特征进行特征生成")
        
        # 创建新特征字典，避免DataFrame碎片化
        new_features_dict = {}
        
        # # 1. PCA降维
        # dim_reducer = DimReducer(reducer_method=reducer_method, reducer_params={'n_components': n_components})
        # dim_reducer.fit(feature_df[config.REMAIN_FEATURES])
        # reduced_feature_df, reduced_cols = dim_reducer.extract(feature_df[config.REMAIN_FEATURES])
        # for col in reduced_cols:
        #     new_features_dict[col] = reduced_feature_df[col]
        # if data_id == list(feature_dict.keys())[0] and oof_file and n_components == 2:
        #     try:
        #         oof_df = pd.read_csv(oof_file, index_col=0)
        #         dim_reducer.plot(reduced_feature_df, oof_df, save_path=f"{config.FIGS_DIR}/dim_reduction_{reducer_method.lower()}.png")
        #         return
        #     except Exception as e:
        #         logger.warning(f"绘制降维可视化失败: {e}")
        # logger.info(f"PCA降维完成，生成 {len(reduced_cols)} 个降维特征")
        
        # 2. 最近邻特征生成
        neighbor_extractor = NeighborFeatureExtractor(metric='euclidean', stage='train')
        neighbor_extractor.fit(feature_df[config.REMAIN_FEATURES], n_neighbors=n_neighbors)
        neighbor_feature_df, neighbor_cols = neighbor_extractor.extract(feature_df[config.REMAIN_FEATURES], neighbor_list=neighbor_list)
        for col in neighbor_cols:
            new_features_dict[col] = neighbor_feature_df[col]
        logger.info(f"最近邻均值特征提取完成，生成 {len(neighbor_cols)} 个特征")
        
        # 3. 一次性创建DataFrame，避免碎片化
        if new_features_dict:
            new_features = pd.DataFrame(new_features_dict, index=feature_df.index)
        else:
            new_features = pd.DataFrame(index=feature_df.index)
        
        if new_features.empty:
            logger.info(f"数据ID '{data_id}' 没有选择任何交互项类型，跳过。")
            updated_feature_dict[data_id] = feature_df.copy()
            continue
        
        logger.info(f"  数据ID '{data_id}' 成功创建 {len(new_features.columns)} 个交互特征")
        
        # 合并特征
        updated_feature_df = feature_df.drop(columns=new_features.columns, errors='ignore')
        updated_feature_df = updated_feature_df.merge(new_features, left_index=True, right_index=True, how='left')
        updated_feature_df = features.clean_feature_names(updated_feature_df, prefix="f_inter")
        
        updated_feature_dict[data_id] = updated_feature_df
        
        # 记录所有交互特征名称（用于元数据）
        if data_id == list(feature_dict.keys())[0]:  # 只记录第一个数据ID的交互特征名称
            all_new_features = new_features.columns.tolist()

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
        
        metadata['last_updated_dim_nn_file'] = base_path.name
        metadata['last_dim_nn_features'] = all_new_features
        metadata['dim_nn_params'] = {
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'neighbor_list': neighbor_list,
            'metric': metric,
            'stage': stage
        }
        metadata['data_ids'] = list(updated_feature_dict.keys())
        
        # 保存结果
        if is_dict_format or len(updated_feature_dict) > 1:
            new_file_path = features._save_feature_dict_file(updated_feature_dict, metadata)
        else:
            # 如果原来是单个DataFrame格式且只有一个数据ID，保持兼容性
            new_file_path = features._save_feature_file(updated_feature_dict["0"], metadata)
        
        logger.info(f"降维和最近邻特征已保存到新文件: {new_file_path.name}")
        
        logger.info("\n=== 降维和最近邻特征生成完成统计 ===")
        for data_id in updated_feature_dict.keys():
            initial_count = initial_feature_counts[data_id]
            final_count = final_feature_counts[data_id]
            logger.info(f"数据ID '{data_id}': {initial_count} -> {final_count} 个特征 (+{final_count - initial_count})")
        
        # 打印新增的特征名列表
        if all_new_features:
            logger.info(f"\n新增的特征列表 (共 {len(all_new_features)} 个):")
            logger.info(f"{all_new_features}")
        else:
            logger.info("\n未生成新的特征")
    else:
        logger.info("没有新特征生成，文件未保存。")
        new_file_path = None
        
