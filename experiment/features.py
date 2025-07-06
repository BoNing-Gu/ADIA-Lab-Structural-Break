import pandas as pd
import numpy as np
import scipy.stats
import statsmodels.tsa.api as tsa
import antropy
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import logging
import os
from pathlib import Path
import inspect
import json

from . import config, utils

logger = logging.getLogger(__name__)

# --- 特征函数注册表 ---
FEATURE_REGISTRY = {}

def register_feature(func):
    """一个用于注册特征函数的装饰器"""
    FEATURE_REGISTRY[func.__name__] = func
    return func

# --- 1. 分布统计特征 ---
@register_feature
def distributional_stats(u: pd.DataFrame) -> dict:
    s1 = u['value'][u['period'] == 0]
    s2 = u['value'][u['period'] == 1]
    feats = {}
    
    mean1, mean2 = s1.mean(), s2.mean()
    feats['mean_diff'] = mean2 - mean1
    
    std1, std2 = s1.std(), s2.std()
    feats['std_diff'] = std2 - std1
    if std1 > 1e-6:
        feats['std_ratio'] = std2 / std1
    else:
        feats['std_ratio'] = 1.0 if std2 < 1e-6 else 1e6
    
    feats['skew_diff'] = s2.skew() - s1.skew()
    feats['kurt_diff'] = s2.kurt() - s1.kurt()
    
    if len(s1) > 1 and len(s2) > 1:
        ks_stat, ks_pvalue = scipy.stats.ks_2samp(s1, s2)
        feats['ks_stat'] = ks_stat
        feats['ks_pvalue'] = -ks_pvalue
    else:
        feats['ks_stat'] = 0
        feats['ks_pvalue'] = 0

    ttest_stat, ttest_pvalue = scipy.stats.ttest_ind(s1, s2, equal_var=False)
    feats['ttest_pvalue'] = -ttest_pvalue if not np.isnan(ttest_pvalue) else 0

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 2. 累积和特征 ---
@register_feature
def cumulative_features(u: pd.DataFrame) -> dict:
    s1 = u['value'][u['period'] == 0]
    s2 = u['value'][u['period'] == 1]
    feats = {}
    
    feats['sum_diff'] = s2.sum() - s1.sum()
    
    if not s1.empty and not s2.empty:
        feats['cumsum_max_diff'] = s2.cumsum().max() - s1.cumsum().max()
    else:
        feats['cumsum_max_diff'] = 0

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 3. 振荡特征 ---
@register_feature
def oscillation_features(u: pd.DataFrame) -> dict:
    s1 = u['value'][u['period'] == 0].reset_index(drop=True)
    s2 = u['value'][u['period'] == 1].reset_index(drop=True)
    feats = {}

    def count_zero_crossings(series: pd.Series):
        if len(series) < 2: return 0
        centered_series = series - series.mean()
        if centered_series.eq(0).all(): return 0
        return np.sum(np.diff(np.sign(centered_series)) != 0)

    feats['zero_cross_diff'] = count_zero_crossings(s2) - count_zero_crossings(s1)
    
    def autocorr_lag1(s):
        if len(s) < 2: return 0.0
        ac = s.autocorr(lag=1)
        return ac if not np.isnan(ac) else 0.0
        
    feats['autocorr_lag1_diff'] = autocorr_lag1(s2) - autocorr_lag1(s1)
    
    feats['diff_var_diff'] = s2.diff().var() - s1.diff().var()
    
    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 4. 周期性特征 ---
@register_feature
def cyclic_features(u: pd.DataFrame) -> dict:
    s1 = u['value'][u['period'] == 0]
    s2 = u['value'][u['period'] == 1]
    feats = {}

    def get_fft_props(series):
        if len(series) < 2: return 0.0, 0.0
        
        N = len(series)
        yf = np.fft.fft(series.values)
        power = np.abs(yf[1:N//2])**2
        xf = np.fft.fftfreq(N, 1)[1:N//2]
        
        if len(power) == 0: return 0.0, 0.0
            
        dominant_freq = xf[np.argmax(power)]
        max_power = np.max(power)
        return dominant_freq, max_power

    freq1, power1 = get_fft_props(s1)
    freq2, power2 = get_fft_props(s2)
    
    feats['dominant_freq_diff'] = freq2 - freq1
    feats['max_power_diff'] = power2 - power1
    
    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 5. 振幅特征 ---
@register_feature
def amplitude_features(u: pd.DataFrame) -> dict:
    s1 = u['value'][u['period'] == 0]
    s2 = u['value'][u['period'] == 1]
    feats = {}
    
    if not s1.empty and not s2.empty:
        feats['ptp_diff'] = np.ptp(s2) - np.ptp(s1)
        feats['iqr_diff'] = scipy.stats.iqr(s2) - scipy.stats.iqr(s1)
    else:
        feats['ptp_diff'] = 0
        feats['iqr_diff'] = 0
    
    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 6. 高阶统计量与非线性趋势变化特征---
@register_feature
def higher_order_stats_features(u: pd.DataFrame) -> dict:
    s1 = u['value'][u['period'] == 0].reset_index(drop=True)
    s2 = u['value'][u['period'] == 1].reset_index(drop=True)
    feats = {}

    def safe_cv(s):
        m = s.mean()
        std = s.std()
        return std / m if abs(m) > 1e-6 else 0.0

    feats['cv_diff'] = safe_cv(s2) - safe_cv(s1)

    def rolling_std_mean(s, window=5):
        if len(s) < window:
            return 0.0
        return s.rolling(window=window).std().dropna().mean()

    feats['rolling_std_diff'] = rolling_std_mean(s2) - rolling_std_mean(s1)

    def slope_theil_sen(s):
        if len(s) < 2:
            return 0.0
        try:
            slope, intercept, _, _ = scipy.stats.theilslopes(s.values, np.arange(len(s)))
            return slope
        except Exception:
            return 0.0

    feats['theil_sen_slope_diff'] = slope_theil_sen(s2) - slope_theil_sen(s1)

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 7. 时间序列建模 ---
@register_feature
def ar_model_residual_features(u: pd.DataFrame) -> dict:
    s1 = u['value'][u['period'] == 0].reset_index(drop=True)
    s2 = u['value'][u['period'] == 1].reset_index(drop=True)
    feats = {}

    def fit_ar(s, lags=10):
        if len(s) <= lags + 1:
            return None
        try:
            return tsa.ar_model.AutoReg(s, lags=lags, old_names=False).fit()
        except Exception:
            return None

    model1 = fit_ar(s1)
    model2 = fit_ar(s2)

    if model1 is not None and model2 is not None:
        feats['ar_resid_std_diff'] = model2.resid.std() - model1.resid.std()
        feats['ar_aic_diff'] = model2.aic - model1.aic
    else:
        feats['ar_resid_std_diff'] = 0.0
        feats['ar_aic_diff'] = 0.0

    if model1 is not None and len(s2) > 0:
        try:
            max_lag = max(model1.model.ar_lags)
            history = s1.iloc[-max_lag:].tolist()
            preds = []

            for t in range(len(s2)):
                lagged_vals = history[-max_lag:]
                pred = model1.params['const'] if 'const' in model1.params else 0.0
                for i, lag in enumerate(model1.model.ar_lags):
                    pred += model1.params[f'value.L{lag}'] * lagged_vals[-lag]
                preds.append(pred)
                history.append(s2.iloc[t])

            preds = np.array(preds)
            mse = np.mean((preds - s2.values[:len(preds)]) ** 2)
            feats['ar_predict_mse'] = mse
        except Exception as e:
            logger.warning(f"AR prediction error: {e}")
            feats['ar_predict_mse'] = 0.0
    else:
        feats['ar_predict_mse'] = 0.0

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 8. 熵信息 ---
@register_feature
def entropy_features(u: pd.DataFrame) -> dict:
    s1 = u['value'][u['period'] == 0].to_numpy()
    s2 = u['value'][u['period'] == 1].to_numpy()
    feats = {}

    def compute_entropy(x):
        hist, _ = np.histogram(x, bins='auto', density=True)
        hist = hist[hist > 0]
        return scipy.stats.entropy(hist)
    feats['shannon_entropy_0'] = compute_entropy(s1)
    feats['shannon_entropy_1'] = compute_entropy(s2)
    feats['shannon_entropy_diff'] = feats['shannon_entropy_1'] - feats['shannon_entropy_0']

    feats['perm_entropy_0'] = antropy.perm_entropy(s1, normalize=True)
    feats['perm_entropy_1'] = antropy.perm_entropy(s2, normalize=True)
    feats['perm_entropy_diff'] = feats['perm_entropy_1'] - feats['perm_entropy_0']

    feats['spectral_entropy_0'] = antropy.spectral_entropy(s1, sf=1.0, normalize=True)
    feats['spectral_entropy_1'] = antropy.spectral_entropy(s2, sf=1.0, normalize=True)
    feats['spectral_entropy_diff'] = feats['spectral_entropy_1'] - feats['spectral_entropy_0']

    feats['svd_entropy_0'] = antropy.svd_entropy(s1, normalize=True)
    feats['svd_entropy_1'] = antropy.svd_entropy(s2, normalize=True)
    feats['svd_entropy_diff'] = feats['svd_entropy_1'] - feats['svd_entropy_0']

    feats['approx_entropy_0'] = antropy.app_entropy(s1)
    feats['approx_entropy_1'] = antropy.app_entropy(s2)
    feats['approx_entropy_diff'] = feats['approx_entropy_1'] - feats['approx_entropy_0']

    feats['sample_entropy_0'] = antropy.sample_entropy(s1)
    feats['sample_entropy_1'] = antropy.sample_entropy(s2)
    feats['sample_entropy_diff'] = feats['sample_entropy_1'] - feats['sample_entropy_0']

    feats['hjorth_mobility_0'], feats['hjorth_complexity_0'] = antropy.hjorth_params(s1)
    feats['hjorth_mobility_1'], feats['hjorth_complexity_1'] = antropy.hjorth_params(s2)
    feats['hjorth_mobility_diff'] = feats['hjorth_mobility_1'] - feats['hjorth_mobility_0']
    feats['hjorth_complexity_diff'] = feats['hjorth_complexity_1'] - feats['hjorth_complexity_0']

    feats['num_zerocross_0'] = antropy.num_zerocross(s1)
    feats['num_zerocross_1'] = antropy.num_zerocross(s2)
    feats['num_zerocross_diff'] = feats['num_zerocross_1'] - feats['num_zerocross_0']

    def series_to_binary_str(x, method='median'):
        if method == 'median':
            threshold = np.median(x)
            return ''.join(['1' if val > threshold else '0' for val in x])
        return None
    bin_str1 = series_to_binary_str(s1)
    bin_str2 = series_to_binary_str(s2)
    feats['lziv_complexity_0'] = antropy.lziv_complexity(bin_str1, normalize=True)
    feats['lziv_complexity_1'] = antropy.lziv_complexity(bin_str2, normalize=True)
    feats['lziv_complexity_diff'] = feats['lziv_complexity_1'] - feats['lziv_complexity_0']

    def estimate_cond_entropy(x, lag=1):
        x = x - np.mean(x)
        x_lag = x[:-lag]
        x_now = x[lag:]
        bins = 10
        joint_hist, _, _ = np.histogram2d(x_lag, x_now, bins=bins, density=True)
        joint_hist = joint_hist[joint_hist > 0]
        H_xy = -np.sum(joint_hist * np.log(joint_hist))
        H_x = -np.sum(np.histogram(x_lag, bins=bins, density=True)[0] * \
                      np.log(np.histogram(x_lag, bins=bins, density=True)[0] + 1e-12))
        return H_xy - H_x
    feats['cond_entropy_0'] = estimate_cond_entropy(s1)
    feats['cond_entropy_1'] = estimate_cond_entropy(s2)
    feats['cond_entropy_diff'] = feats['cond_entropy_1'] - feats['cond_entropy_0']
    
    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 9. 分形 ---
@register_feature
def fractal_dimension_features(u: pd.DataFrame) -> dict:
    s1 = u['value'][u['period'] == 0].to_numpy()
    s2 = u['value'][u['period'] == 1].to_numpy()
    feats = {}

    feats['petrosian_fd_diff'] = (antropy.petrosian_fd(s1) - antropy.petrosian_fd(s2)) * 100
    feats['katz_fd_diff'] = (antropy.katz_fd(s1) - antropy.katz_fd(s2)) * 10
    feats['higuchi_fd_diff'] = (antropy.higuchi_fd(s1) - antropy.higuchi_fd(s2)) * 100
    feats['detrended_fluctuation_diff'] = (antropy.detrended_fluctuation(s1) - antropy.detrended_fluctuation(s2)) * 10

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 特征管理核心逻辑 ---

def _load_feature_file():
    """加载特征文件及其元数据。如果文件不存在，则返回空的 DataFrame 和元数据。"""
    if not config.FEATURE_FILE.exists():
        return pd.DataFrame(), {}
    
    try:
        table = pd.read_parquet(config.FEATURE_FILE)
        metadata_str = table.attrs.get(b'feature_metadata', b'{}')
        metadata = json.loads(metadata_str.decode('utf-8'))
        return table, metadata
    except Exception as e:
        logger.warning(f"无法加载特征文件或元数据: {e}。将创建一个新的特征文件。")
        return pd.DataFrame(), {}

def _save_feature_file(df: pd.DataFrame, metadata: dict):
    """保存特征文件及其元数据。"""
    df.attrs['feature_metadata'] = json.dumps(metadata)
    df.to_parquet(config.FEATURE_FILE)

def _backup_feature_file():
    """如果特征文件存在，则备份它。"""
    if config.FEATURE_FILE.exists():
        config.FEATURE_BACKUP_DIR.mkdir(exist_ok=True, parents=True)
        timestamp = utils.get_timestamp()
        backup_path = config.FEATURE_BACKUP_DIR / f'features_{timestamp}.parquet'
        config.FEATURE_FILE.rename(backup_path)
        logger.info(f"已将旧特征文件备份到: {backup_path}")

def _apply_feature_func_parallel(func, X_df: pd.DataFrame) -> pd.DataFrame:
    """并行应用单个特征函数"""
    all_ids = X_df.index.get_level_values("id").unique()
    results = Parallel(n_jobs=-1)(
        delayed(lambda df_id, id_val: {**{'id': id_val}, **func(df_id)})(X_df.loc[id_val], id_val)
        for id_val in tqdm(all_ids, desc=f"Running {func.__name__}")
    )
    return pd.DataFrame(results).set_index('id')

def generate_features(X_df: pd.DataFrame, funcs_to_run: list = None):
    """
    生成指定的特征，并更新持久化的特征文件。
    - 如果 funcs_to_run 为 None，则生成所有已注册的特征。
    - 如果 funcs_to_run 不为 None，则只生成/更新列表中的特征。
    """
    if funcs_to_run is None:
        funcs_to_run = list(FEATURE_REGISTRY.keys())
        logger.info("将生成所有已注册的特征。")
    else:
        logger.info(f"将生成指定的特征: {funcs_to_run}")

    # 1. 备份并加载现有特征
    _backup_feature_file()
    feature_df, metadata = _load_feature_file()

    # 2. 确定要运行的函数
    valid_funcs = [f for f in funcs_to_run if f in FEATURE_REGISTRY]
    if len(valid_funcs) != len(funcs_to_run):
        invalid_funcs = set(funcs_to_run) - set(valid_funcs)
        logger.warning(f"以下特征函数未在注册表中找到，将被忽略: {invalid_funcs}")

    # 3. 逐个生成新特征并更新
    for func_name in valid_funcs:
        func = FEATURE_REGISTRY[func_name]
        logger.info(f"--- 开始生成特征: {func_name} ---")
        
        # 3.1 如果该函数之前生成过特征，先从主DataFrame中删除它们
        if func_name in metadata:
            cols_to_drop = [col for col in metadata[func_name] if col in feature_df.columns]
            if cols_to_drop:
                feature_df.drop(columns=cols_to_drop, inplace=True)
                logger.info(f"已删除旧版特征: {cols_to_drop}")

        # 3.2 生成新特征
        new_feature_df = _apply_feature_func_parallel(func, X_df)
        
        # 3.3 合并新特征并更新元数据
        feature_df = feature_df.join(new_feature_df, how='outer')
        metadata[func_name] = new_feature_df.columns.tolist()
        logger.info(f"--- 完成生成特征: {func_name} ---")

    # 4. 保存更新后的文件
    _save_feature_file(feature_df, metadata)
    logger.info(f"特征文件已更新。当前总特征数: {len(feature_df.columns)}")
    return feature_df

def delete_features(funcs_to_delete: list):
    """
    从持久化的特征文件中删除指定的特征。
    """
    if not funcs_to_delete:
        logger.warning("未指定任何要删除的特征。")
        return

    logger.info(f"将删除以下函数对应的特征: {funcs_to_delete}")

    # 1. 备份并加载
    _backup_feature_file()
    feature_df, metadata = _load_feature_file()

    if feature_df.empty:
        logger.warning("特征文件为空或不存在，无需删除。")
        return

    # 2. 确定要删除的列
    cols_to_drop = []
    for func_name in funcs_to_delete:
        if func_name in metadata:
            cols_to_drop.extend(metadata[func_name])
            del metadata[func_name] # 从元数据中移除
        else:
            logger.warning(f"函数 {func_name} 未在元数据中找到，可能之前未生成过或已被删除。")
    
    # 3. 删除列
    final_cols_to_drop = [col for col in cols_to_drop if col in feature_df.columns]
    if final_cols_to_drop:
        feature_df.drop(columns=final_cols_to_drop, inplace=True)
        logger.info(f"已成功删除特征: {final_cols_to_drop}")
    else:
        logger.info("没有需要删除的特征列。")

    # 4. 保存
    _save_feature_file(feature_df, metadata)
    logger.info(f"特征文件已更新。当前总特征数: {len(feature_df.columns)}")

def load_features() -> pd.DataFrame:
    """加载持久化的特征文件。"""
    logger.info(f"正在从 {config.FEATURE_FILE} 加载特征...")
    feature_df, _ = _load_feature_file()
    if feature_df.empty:
        logger.error("特征文件为空或不存在！请先生成特征。")
        raise FileNotFoundError("特征文件未找到或为空。")
    logger.info(f"特征加载成功，共 {len(feature_df.columns)} 个特征。")
    return feature_df 