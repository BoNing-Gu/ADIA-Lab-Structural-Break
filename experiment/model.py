import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

class DimReducer:
    def __init__(self, seg='left', reducer=None, reducer_params=None):
        """
        seg: 'left' or 'right'，指定使用特征名中带有哪个后缀的特征
        reducer: sklearn 降维器类（不实例化）
        reducer_params: dict，降维器的初始化参数
        """
        assert seg in ['left', 'right'], "seg must be 'left' or 'right'"
        self.seg = seg
        self.reducer_class = reducer if reducer is not None else PCA
        self.reducer_params = reducer_params if reducer_params is not None else {'n_components': 2}
        self.model = None
        self.feature_names_ = None  # 记录参与降维的特征名

    def fit(self, X_train):
        # 选出带有后缀的特征列
        cols = [col for col in X_train.columns if col.endswith(f'_{self.seg}')]
        if len(cols) == 0:
            raise ValueError(f"No features with suffix _{self.seg} found.")
        self.feature_names_ = cols
        self.model = self.reducer_class(**self.reducer_params)
        self.model.fit(X_train[cols].values)

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
        reduced = self.model.transform(feature_df[self.feature_names_].values)
        reduced_cols = [f'dim_{i}_{self.seg}' for i in range(reduced.shape[1])]
        reduced_df = pd.DataFrame(reduced, columns=reduced_cols, index=feature_df.index)
        feature_df = pd.concat([feature_df, reduced_df], axis=1)
        return feature_df, reduced_cols

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'seg': self.seg,
                'reducer_class': self.reducer_class,
                'reducer_params': self.reducer_params,
                'model': self.model,
                'feature_names_': self.feature_names_
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        self.seg = obj['seg']
        self.reducer_class = obj['reducer_class']
        self.reducer_params = obj['reducer_params']
        self.model = obj['model']
        self.feature_names_ = obj['feature_names_']

class NeighborFeatureExtractor:
    def __init__(self, metric='euclidean', stage='train'):
        self.metric = metric
        self.nn_model = None
        self.X_train = None
        self.y_train = None
        self.stage = stage
        if self.stage == 'train':
            self.add_n_neighbors = 1
        else:
            self.add_n_neighbors = 0

    def fit(self, X_train, y_train, n_neighbors=50):
        """
        X_train: pd.DataFrame
        y_train: pd.Series
        n_neighbors: 最大邻居数（内部存储）
        """
        self.X_train = X_train.copy()
        self.y_train = y_train.copy().reset_index(drop=True)
        self.nn_model = NearestNeighbors(
            n_neighbors=n_neighbors + self.add_n_neighbors,
            metric=self.metric
        )
        self.nn_model.fit(self.X_train.values)

    def predict(self, x_query, neighbor_list=[3, 5, 7, 10]):
        """
        x_query: 1D array-like, shape (n_features,)
        neighbor_list: list of ints, 每个值是希望提取的邻居个数
        返回：dict，形如 {'nn_3_mean': xxx, 'nn_5_mean': xxx, ...}
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
                feats[f'nn_{k}_mean'] = np.nan
            else:
                neighbor_idx = neighbor_indices[:k]
                neighbor_y = self.y_train.iloc[neighbor_idx].values
                feats[f'nn_{k}_mean'] = neighbor_y.mean()
        return feats

    def extract(self, feature_df, neighbor_list=[3, 5, 7, 10]):
        feature_df = feature_df.copy()
        extracted_feature_df = feature_df.apply(
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
                'y_train': self.y_train,
                'nn_model': self.nn_model
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        self.metric = obj['metric']
        self.X_train = obj['X_train']
        self.y_train = obj['y_train']
        self.nn_model = obj['nn_model']