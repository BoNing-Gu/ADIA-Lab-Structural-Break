import pandas as pd
import numpy as np
import scipy.stats
import statsmodels.tsa.api as tsa
from statsmodels.tsa.ar_model import AutoReg
import antropy
from tsfresh.feature_extraction import feature_calculators as tsfresh_fe
import ruptures as rpt

import re
import json
import time
import logging
import inspect
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
from joblib import Parallel, delayed
from typing import List, Dict, Tuple, Optional
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
import os
from . import config, utils

# --- GPU 加速配置 ---
# 尝试导入GPU相关库并检查可用性
try:
    import cudf
    import cupy
    from numba import cuda
    GPU_AVAILABLE = cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
# --- END ---

os.environ["STUMPY_GPU"] = "0" 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", InterpolationWarning)
warnings.simplefilter("ignore", FutureWarning)

# logger 将由 main.py 在运行时注入
logger = None

# --- 特征函数注册表 ---
FEATURE_REGISTRY = {}

def register_feature(_func=None, *, parallelizable=True, func_id=""):
    """一个用于注册特征函数的装饰器，可以标记特征是否可并行化。"""
    def decorator_register(func):
        FEATURE_REGISTRY[func.__name__] = {
            "func": func, 
            "parallelizable": parallelizable,
            "func_id": func_id
        }
        return func

    if _func is None:
        # Used as @register_feature(parallelizable=...)
        return decorator_register
    else:
        # Used as @register_feature
        return decorator_register(_func)

def _add_contribution_ratio_feats(feats: dict, name: str, left, right, whole):
    """
    一个辅助函数，用于向特征字典中添加贡献度和与整体的比例特征。

    Args:
        feats (dict): 要更新的特征字典。
        name (str): 特征的基础名称 (例如, 'stats_mean')。
        left (float): 左侧分段的特征值。
        right (float): 右侧分段的特征值。
        whole (float): 整个序列的特征值。
    """
    # np.isnan for both None and np.nan
    if np.isnan(left) or np.isnan(right) or np.isnan(whole) or whole is None or left is None or right is None:
        return
        
    # 特征贡献度
    feats[f'{name}_contribution_left'] = left / (left + right + 1e-6)
    feats[f'{name}_contribution_right'] = right / (left + right + 1e-6)
    
    # 与整体特征的比例
    feats[f'{name}_ratio_to_whole_left'] = left / (whole + 1e-6)
    feats[f'{name}_ratio_to_whole_right'] = right / (whole + 1e-6)

# --- 1. 分布统计特征 ---
def safe_cv(s):
    s = pd.Series(s)
    m = s.mean()
    std = s.std()
    return std / m if abs(m) > 1e-6 else 0.0

def rolling_std_mean(s, window=5):
    s = pd.Series(s)
    if len(s) < window:
        return 0.0
    return s.rolling(window=window).std().dropna().mean()

def slope_theil_sen(s):
    s = pd.Series(s)
    if len(s) < 2:
        return 0.0
    try:
        slope, intercept, _, _ = scipy.stats.theilslopes(s.values, np.arange(len(s)))
        return slope
    except Exception:
        return 0.0

class STATSFeatureExtractor:
    def __init__(self):
        # 所有可用的func类及其名称
        self.func_classes = {
            'mean': np.mean,
            'median': np.median,
            'max': np.max,
            'min': np.min,
            'range': lambda x: np.max(x) - np.min(x),
            'std': np.std,
            'skew': scipy.stats.skew,
            'kurt': scipy.stats.kurtosis,
            'cv': safe_cv,
            'mean_of_rolling_std': rolling_std_mean,
            'theil_sen_slope': slope_theil_sen
        }
    
    def fit(self, signal):
        self.signal = np.asarray(signal)
        self.n = len(signal)

    def calculate(self, func, start, end):
        result = func(self.signal[start:end])
        if isinstance(result, float) or isinstance(result, int):
            return result
        else:
            return result.item()

    def extract(self, signal, boundary):
        """
        输入：
            signal: 1D numpy array，单变量时间序列
            boundary: int，分割点
        输出：
            result: dict，格式为 {func_name: {'left': value, 'right': value}}
        """
        n = self.n
        result = {}
        for name, func in self.func_classes.items():
            try:
                left = self.calculate(func, 0, boundary)
                right = self.calculate(func, boundary, n)
                whole = self.calculate(func, 0, n)
                diff = right - left
                ratio = right / (left + 1e-6)
            except Exception:
                left = None
                right = None
                whole = None
                diff = None
                ratio = None
            result[name] = {'left': left, 'right': right, 'whole': whole, 'diff': diff, 'ratio': ratio}
        return result

@register_feature(func_id="1")
def distribution_stats_features(u: pd.DataFrame) -> dict:
    """统计量的分段值、Diff值、Ratio值"""
    value = u['value'].values.astype(np.float32)
    period = u['period'].values.astype(np.float32)
    boundary = np.where(np.diff(period) != 0)[0].item()
    feats = {}

    extractor = STATSFeatureExtractor()
    extractor.fit(value)
    features = extractor.extract(value, boundary)

    feats = {}
    for k, v in features.items():
        for seg, value in v.items():
            feats[f'stats_{k}_{seg}'] = value
        _add_contribution_ratio_feats(feats, f'stats_{k}', v['left'], v['right'], v['whole'])

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}
    
# --- 2. 假设检验统计量特征 ---
@register_feature(func_id="2")
def test_stats_features(u: pd.DataFrame) -> dict:
    s1 = u['value'][u['period'] == 0]
    s2 = u['value'][u['period'] == 1]
    s_whole = u['value']
    feats = {}

    """假设检验统计量"""
    # KS检验
    ks_stat, ks_pvalue = scipy.stats.ks_2samp(s1, s2)
    feats['ks_stat'] = ks_stat
    feats['ks_pvalue'] = -ks_pvalue

    # T检验
    ttest_stat, ttest_pvalue = scipy.stats.ttest_ind(s1, s2, equal_var=False)
    feats['ttest_pvalue'] = -ttest_pvalue if not np.isnan(ttest_pvalue) else 1

    # AD检验
    ad_stat, _, ad_pvalue = scipy.stats.anderson_ksamp([s1.to_numpy(), s2.to_numpy()])
    feats['ad_stat'] = ad_stat
    feats['ad_pvalue'] = -ad_pvalue

    # Mann-Whitney U检验 (非参数，不假设分布)
    mw_stat, mw_pvalue = scipy.stats.mannwhitneyu(s1, s2, alternative='two-sided')
    feats['mannwhitney_stat'] = mw_stat if not np.isnan(mw_stat) else 0
    feats['mannwhitney_pvalue'] = -mw_pvalue if not np.isnan(mw_pvalue) else 1
    
    # Wilcoxon秩和检验
    try:
        w_stat, w_pvalue = scipy.stats.ranksums(s1, s2)
        feats['wilcoxon_stat'] = w_stat if not np.isnan(w_stat) else 0
        feats['wilcoxon_pvalue'] = -w_pvalue if not np.isnan(w_pvalue) else 1
    except ValueError:
        feats['wilcoxon_stat'] = 0
        feats['wilcoxon_pvalue'] = 1

    # Levene检验
    levene_stat, levene_pvalue = scipy.stats.levene(s1, s2)
    feats['levene_stat'] = levene_stat if not np.isnan(levene_stat) else 0
    feats['levene_pvalue'] = -levene_pvalue if not np.isnan(levene_pvalue) else 1
    
    # Bartlett检验
    bartlett_stat, bartlett_pvalue = scipy.stats.bartlett(s1, s2)
    feats['bartlett_stat'] = bartlett_stat if not np.isnan(bartlett_stat) else 0
    feats['bartlett_pvalue'] = -bartlett_pvalue if not np.isnan(bartlett_pvalue) else 1
    
    """分段假设检验的分段值、Diff值、Ratio值"""
    # Shapiro-Wilk检验
    sw1_stat, sw1_pvalue, sw2_stat, sw2_pvalue, sw_whole_stat, sw_whole_pvalue = (np.nan,)*6
    if len(s1) <= 5000 and len(s1) > 2:
        sw1_stat, sw1_pvalue = scipy.stats.shapiro(s1)
    if len(s2) <= 5000 and len(s2) > 2:
        sw2_stat, sw2_pvalue = scipy.stats.shapiro(s2)
    if len(s_whole) <= 5000 and len(s_whole) > 2:
        sw_whole_stat, sw_whole_pvalue = scipy.stats.shapiro(s_whole)
    
    feats['shapiro_pvalue_left'] = sw1_pvalue
    feats['shapiro_pvalue_right'] = sw2_pvalue
    feats['shapiro_pvalue_whole'] = sw_whole_pvalue
    feats['shapiro_pvalue_diff'] = sw2_pvalue - sw1_pvalue if not (np.isnan(sw1_pvalue) or np.isnan(sw2_pvalue)) else 0
    feats['shapiro_pvalue_ratio'] = sw2_pvalue / (sw1_pvalue + 1e-6) if not (np.isnan(sw1_pvalue) or np.isnan(sw2_pvalue)) else 0
    _add_contribution_ratio_feats(feats, 'shapiro_pvalue', sw1_pvalue, sw2_pvalue, sw_whole_pvalue)

    # Jarque-Bera检验差异
    jb1_stat, jb1_pvalue, jb2_stat, jb2_pvalue, jb_whole_stat, jb_whole_pvalue = (np.nan,)*6
    try:
        if len(s1) > 2: jb1_stat, jb1_pvalue = scipy.stats.jarque_bera(s1)
        if len(s2) > 2: jb2_stat, jb2_pvalue = scipy.stats.jarque_bera(s2)
        if len(s_whole) > 2: jb_whole_stat, jb_whole_pvalue = scipy.stats.jarque_bera(s_whole)
    except:
        pass
    
    feats['jb_pvalue_left'] = jb1_pvalue
    feats['jb_pvalue_right'] = jb2_pvalue
    feats['jb_pvalue_whole'] = jb_whole_pvalue
    feats['jb_pvalue_diff'] = jb2_pvalue - jb1_pvalue if not (np.isnan(jb1_pvalue) or np.isnan(jb2_pvalue)) else 0
    feats['jb_pvalue_ratio'] = jb2_pvalue / (jb1_pvalue + 1e-6) if not (np.isnan(jb1_pvalue) or np.isnan(jb2_pvalue)) else 0
    _add_contribution_ratio_feats(feats, 'jb_pvalue', jb1_pvalue, jb2_pvalue, jb_whole_pvalue)

    # KPSS检验
    def extract_kpss_features(s):
        if len(s) <= 12:
            return {'p': 0.1, 'stat': 0.0, 'lag': 0, 'crit_5pct': 0.0, 'reject_5pct': 0}
        kpss = tsa.stattools.kpss(s, regression='c', nlags='auto')
        stat, p, lag, crit = kpss
        crit_5pct = crit['5%']
        return {
            'p': p,
            'stat': stat,
            'lag': lag,
            'crit_5pct': crit_5pct,
            'reject_5pct': int(stat > crit_5pct)  # KPSS原假设是"平稳"，所以 > 临界值 拒绝平稳
        }
    try:
        k1 = extract_kpss_features(s1)
        k2 = extract_kpss_features(s2)
        k_whole = extract_kpss_features(s_whole)

        feats['kpss_pvalue_left'] = k1['p']
        feats['kpss_pvalue_right'] = k2['p']
        feats['kpss_pvalue_whole'] = k_whole['p']
        feats['kpss_pvalue_diff'] = k2['p'] - k1['p']
        feats['kpss_pvalue_ratio'] = k2['p'] / (k1['p'] + 1e-6)
        _add_contribution_ratio_feats(feats, 'kpss_pvalue', k1['p'], k2['p'], k_whole['p'])

        feats['kpss_stat_left'] = k1['stat']
        feats['kpss_stat_right'] = k2['stat']
        feats['kpss_stat_whole'] = k_whole['stat']
        feats['kpss_stat_diff'] = k2['stat'] - k1['stat']
        feats['kpss_stat_ratio'] = k2['stat'] / (k1['stat'] + 1e-6)
        _add_contribution_ratio_feats(feats, 'kpss_stat', k1['stat'], k2['stat'], k_whole['stat'])
    except:
        feats.update({
            'kpss_pvalue_left': 1, 'kpss_pvalue_right': 1, 'kpss_pvalue_whole': 1, 'kpss_pvalue_diff': 0, 'kpss_pvalue_ratio': 0,
            'kpss_stat_left': 0, 'kpss_stat_right': 0, 'kpss_stat_whole': 0, 'kpss_stat_diff': 0, 'kpss_stat_ratio': 0
        })

    # 平稳性检验 (ADF)
    def extract_adf_features(s):
        if len(s) <= 12:
            return {'p': 1.0, 'stat': 0.0, 'lag': 0, 'ic': 0.0, 'crit_5pct': 0.0, 'reject_5pct': 0}
        adf = tsa.stattools.adfuller(s, autolag='AIC')
        stat, p, lag, _, crit, ic = adf
        crit_5pct = crit['5%']
        return {
            'p': p,
            'stat': stat,
            'lag': lag,
            'ic': ic,
            'crit_5pct': crit_5pct,
            'reject_5pct': int(stat < crit_5pct)
        }
    try:
        f1 = extract_adf_features(s1)
        f2 = extract_adf_features(s2)
        f_whole = extract_adf_features(s_whole)

        feats['adf_pvalue_left'] = f1['p']
        feats['adf_pvalue_right'] = f2['p']
        feats['adf_pvalue_whole'] = f_whole['p']
        feats['adf_pvalue_diff'] = f2['p'] - f1['p']
        feats['adf_pvalue_ratio'] = f2['p'] / (f1['p'] + 1e-6)
        _add_contribution_ratio_feats(feats, 'adf_pvalue', f1['p'], f2['p'], f_whole['p'])

        feats['adf_stat_left'] = f1['stat']
        feats['adf_stat_right'] = f2['stat']
        feats['adf_stat_whole'] = f_whole['stat']
        feats['adf_stat_diff'] = f2['stat'] - f1['stat']
        feats['adf_stat_ratio'] = f2['stat'] / (f1['stat'] + 1e-6)
        _add_contribution_ratio_feats(feats, 'adf_stat', f1['stat'], f2['stat'], f_whole['stat'])

        feats['adf_icbest_left'] = f1['ic']
        feats['adf_icbest_right'] = f2['ic']
        feats['adf_icbest_whole'] = f_whole['ic']
        feats['adf_icbest_diff'] = f2['ic'] - f1['ic']
        feats['adf_icbest_ratio'] = f2['ic'] / (f1['ic'] + 1e-6)
        _add_contribution_ratio_feats(feats, 'adf_icbest', f1['ic'], f2['ic'], f_whole['ic'])
    except:
        feats.update({
            'adf_pvalue_left': 1, 'adf_pvalue_right': 1, 'adf_pvalue_whole': 1, 'adf_pvalue_diff': 0, 'adf_pvalue_ratio': 0,
            'adf_stat_left': 0, 'adf_stat_right': 0, 'adf_stat_whole': 0, 'adf_stat_diff': 0, 'adf_stat_ratio': 0,
            'adf_icbest_left': 0, 'adf_icbest_right': 0, 'adf_icbest_whole': 0, 'adf_icbest_diff': 0, 'adf_icbest_ratio': 0
        })


    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 3. 累积和特征 ---
@register_feature(func_id="3")
def cumulative_features(u: pd.DataFrame) -> dict:
    s1 = u['value'][u['period'] == 0]
    s2 = u['value'][u['period'] == 1]
    s_whole = u['value']
    feats = {}
    
    sum1, sum2, sum_whole = s1.sum(), s2.sum(), s_whole.sum()
    feats['sum_left'] = sum1
    feats['sum_right'] = sum2
    feats['sum_whole'] = sum_whole
    feats['sum_diff'] = sum2 - sum1
    feats['sum_ratio'] = sum2 / (sum1 + 1e-6)
    _add_contribution_ratio_feats(feats, 'sum', sum1, sum2, sum_whole)
    
    cumsum1_max = s1.cumsum().max()
    cumsum2_max = s2.cumsum().max()
    cumsum_whole_max = s_whole.cumsum().max()
    feats['cumsum_max_left'] = cumsum1_max
    feats['cumsum_max_right'] = cumsum2_max
    feats['cumsum_max_whole'] = cumsum_whole_max
    feats['cumsum_max_diff'] = cumsum2_max - cumsum1_max
    feats['cumsum_max_ratio'] = cumsum2_max / (cumsum1_max + 1e-6)
    _add_contribution_ratio_feats(feats, 'cumsum_max', cumsum1_max, cumsum2_max, cumsum_whole_max)

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 4. 振荡特征 ---
@register_feature(func_id="4")
def oscillation_features(u: pd.DataFrame) -> dict:
    s1 = u['value'][u['period'] == 0].reset_index(drop=True)
    s2 = u['value'][u['period'] == 1].reset_index(drop=True)
    s_whole = u['value'].reset_index(drop=True)
    feats = {}

    def count_zero_crossings(series: pd.Series):
        if len(series) < 2: return 0
        centered_series = series - series.mean()
        if centered_series.eq(0).all(): return 0
        return np.sum(np.diff(np.sign(centered_series)) != 0)

    zc1, zc2, zc_whole = count_zero_crossings(s1), count_zero_crossings(s2), count_zero_crossings(s_whole)
    feats['zero_cross_left'] = zc1
    feats['zero_cross_right'] = zc2
    feats['zero_cross_whole'] = zc_whole
    feats['zero_cross_diff'] = zc2 - zc1
    feats['zero_cross_ratio'] = zc2 / (zc1 + 1e-6)
    _add_contribution_ratio_feats(feats, 'zero_cross', zc1, zc2, zc_whole)
    
    def autocorr_lag1(s):
        if len(s) < 2: return 0.0
        ac = s.autocorr(lag=1)
        return ac if not np.isnan(ac) else 0.0
        
    ac1, ac2, ac_whole = autocorr_lag1(s1), autocorr_lag1(s2), autocorr_lag1(s_whole)
    feats['autocorr_lag1_left'] = ac1
    feats['autocorr_lag1_right'] = ac2
    feats['autocorr_lag1_whole'] = ac_whole
    feats['autocorr_lag1_diff'] = ac2 - ac1
    feats['autocorr_lag1_ratio'] = ac2 / (ac1 + 1e-6)
    _add_contribution_ratio_feats(feats, 'autocorr_lag1', ac1, ac2, ac_whole)

    var1, var2, var_whole = s1.diff().var(), s2.diff().var(), s_whole.diff().var()
    feats['diff_var_left'] = var1
    feats['diff_var_right'] = var2
    feats['diff_var_whole'] = var_whole
    feats['diff_var_diff'] = var2 - var1
    feats['diff_var_ratio'] = var2 / (var1 + 1e-6)
    _add_contribution_ratio_feats(feats, 'diff_var', var1, var2, var_whole)
    
    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 5. 周期性特征 ---
@register_feature(func_id="5")
def cyclic_features(u: pd.DataFrame) -> dict:
    s1 = u['value'][u['period'] == 0]
    s2 = u['value'][u['period'] == 1]
    s_whole = u['value']
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
    freq_whole, power_whole = get_fft_props(s_whole)
    
    feats['dominant_freq_left'] = freq1
    feats['dominant_freq_right'] = freq2
    feats['dominant_freq_whole'] = freq_whole
    feats['dominant_freq_diff'] = freq2 - freq1
    feats['dominant_freq_ratio'] = freq2 / (freq1 + 1e-6)
    _add_contribution_ratio_feats(feats, 'dominant_freq', freq1, freq2, freq_whole)

    feats['max_power_left'] = power1
    feats['max_power_right'] = power2
    feats['max_power_whole'] = power_whole
    feats['max_power_diff'] = power2 - power1
    feats['max_power_ratio'] = power2 / (power1 + 1e-6)
    _add_contribution_ratio_feats(feats, 'max_power', power1, power2, power_whole)
    
    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 6. 振幅特征 ---
@register_feature(func_id="6")
def amplitude_features(u: pd.DataFrame) -> dict:
    s1 = u['value'][u['period'] == 0]
    s2 = u['value'][u['period'] == 1]
    s_whole = u['value']
    feats = {}
    
    ptp1, ptp2, ptp_whole = np.ptp(s1), np.ptp(s2), np.ptp(s_whole)
    iqr1, iqr2, iqr_whole = scipy.stats.iqr(s1), scipy.stats.iqr(s2), scipy.stats.iqr(s_whole)

    feats['ptp_left'] = ptp1
    feats['ptp_right'] = ptp2
    feats['ptp_whole'] = ptp_whole
    feats['ptp_diff'] = ptp2 - ptp1
    feats['ptp_ratio'] = ptp2 / (ptp1 + 1e-6)
    _add_contribution_ratio_feats(feats, 'ptp', ptp1, ptp2, ptp_whole)

    feats['iqr_left'] = iqr1
    feats['iqr_right'] = iqr2
    feats['iqr_whole'] = iqr_whole
    feats['iqr_diff'] = iqr2 - iqr1
    feats['iqr_ratio'] = iqr2 / (iqr1 + 1e-6)
    _add_contribution_ratio_feats(feats, 'iqr', iqr1, iqr2, iqr_whole)
    
    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 7. 波动性的波动性特征 ---
@register_feature(func_id="7")
def volatility_of_volatility_features(u: pd.DataFrame) -> dict:
    """
    计算滚动标准差序列的统计特征，以捕捉"波动性的波动性"的变化。
    在Period 0和Period 1内部，分别计算小窗口（如长度为50）的滚动标准差，
    然后比较这两条新的滚动标准差序列的均值，生成四个相关特征：
    1. Period 0 的滚动标准差均值
    2. Period 1 的滚动标准差均值
    3. 两者之差
    4. 两者之比
    """
    s1 = u['value'][u['period'] == 0]
    s2 = u['value'][u['period'] == 1]
    s_whole = u['value']
    feats = {}
    window = 50

    def get_rolling_std_mean(s, w):
        if len(s) < w:
            return 0.0
        rolling_std = s.rolling(window=w).std().dropna()
        if rolling_std.empty:
            return 0.0
        return rolling_std.mean()

    mean1 = get_rolling_std_mean(s1, window)
    mean2 = get_rolling_std_mean(s2, window)
    mean_whole = get_rolling_std_mean(s_whole, window)

    feats[f'rolling_std_w{window}_mean_left'] = mean1
    feats[f'rolling_std_w{window}_mean_right'] = mean2
    feats[f'rolling_std_w{window}_mean_whole'] = mean_whole
    feats[f'rolling_std_w{window}_mean_diff'] = mean2 - mean1
    feats[f'rolling_std_w{window}_mean_ratio'] = mean2 / (mean1 + 1e-6)
    _add_contribution_ratio_feats(feats, f'rolling_std_w{window}_mean', mean1, mean2, mean_whole)
    
    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 8. 熵信息 ---
@register_feature(func_id="8")
def entropy_features(u: pd.DataFrame) -> dict:
    s1 = u['value'][u['period'] == 0].to_numpy()
    s2 = u['value'][u['period'] == 1].to_numpy()
    s_whole = u['value'].to_numpy()
    feats = {}

    def compute_entropy(x):
        hist, _ = np.histogram(x, bins='auto', density=True)
        hist = hist[hist > 0]
        return scipy.stats.entropy(hist)
    
    entropy_funcs = {
        'shannon_entropy': compute_entropy,
        'perm_entropy': lambda x: antropy.perm_entropy(x, normalize=True),
        'spectral_entropy': lambda x: antropy.spectral_entropy(x, sf=1.0, normalize=True),
        'svd_entropy': lambda x: antropy.svd_entropy(x, normalize=True),
        'approx_entropy': antropy.app_entropy,
        'sample_entropy': antropy.sample_entropy,
        'petrosian_fd': antropy.petrosian_fd,
        'katz_fd': antropy.katz_fd,
        'higuchi_fd': antropy.higuchi_fd,
        'detrended_fluctuation': antropy.detrended_fluctuation,
    }

    for name, func in entropy_funcs.items():
        try:
            v1, v2, v_whole = func(s1), func(s2), func(s_whole)
            feats[f'{name}_left'] = v1
            feats[f'{name}_right'] = v2
            feats[f'{name}_whole'] = v_whole
            feats[f'{name}_diff'] = v2 - v1
            feats[f'{name}_ratio'] = v2 / (v1 + 1e-6)
            _add_contribution_ratio_feats(feats, name, v1, v2, v_whole)
        except Exception:
            feats.update({f'{name}_left': 0, f'{name}_right': 0, f'{name}_whole': 0, f'{name}_diff': 0, f'{name}_ratio': 0})

    try:
        m1, c1 = antropy.hjorth_params(s1)
        m2, c2 = antropy.hjorth_params(s2)
        m_whole, c_whole = antropy.hjorth_params(s_whole)
        feats.update({
            'hjorth_mobility_left': m1, 'hjorth_mobility_right': m2, 'hjorth_mobility_whole': m_whole,
            'hjorth_mobility_diff': m2 - m1, 'hjorth_mobility_ratio': m2 / (m1 + 1e-6),
            'hjorth_complexity_left': c1, 'hjorth_complexity_right': c2, 'hjorth_complexity_whole': c_whole,
            'hjorth_complexity_diff': c2 - c1, 'hjorth_complexity_ratio': c2 / (c1 + 1e-6)
        })
        _add_contribution_ratio_feats(feats, 'hjorth_mobility', m1, m2, m_whole)
        _add_contribution_ratio_feats(feats, 'hjorth_complexity', c1, c2, c_whole)
    except Exception:
        feats.update({'hjorth_mobility_left':0, 'hjorth_mobility_right':0, 'hjorth_mobility_whole':0, 'hjorth_mobility_diff':0, 'hjorth_mobility_ratio':0,
                     'hjorth_complexity_left':0, 'hjorth_complexity_right':0, 'hjorth_complexity_whole':0, 'hjorth_complexity_diff':0, 'hjorth_complexity_ratio':0})


    def series_to_binary_str(x, method='median'):
        if method == 'median':
            threshold = np.median(x)
            return ''.join(['1' if val > threshold else '0' for val in x])
        return None
    bin_str1 = series_to_binary_str(s1)
    bin_str2 = series_to_binary_str(s2)
    bin_str_whole = series_to_binary_str(s_whole)

    try:
        lz1, lz2, lz_whole = antropy.lziv_complexity(bin_str1, normalize=True), antropy.lziv_complexity(bin_str2, normalize=True), antropy.lziv_complexity(bin_str_whole, normalize=True)
        feats.update({
            'lziv_complexity_left': lz1, 'lziv_complexity_right': lz2, 'lziv_complexity_whole': lz_whole,
            'lziv_complexity_diff': lz2 - lz1, 'lziv_complexity_ratio': lz2 / (lz1 + 1e-6)
        })
        _add_contribution_ratio_feats(feats, 'lziv_complexity', lz1, lz2, lz_whole)
    except Exception:
        feats.update({'lziv_complexity_left':0, 'lziv_complexity_right':0, 'lziv_complexity_whole':0, 'lziv_complexity_diff':0, 'lziv_complexity_ratio':0})


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
    try:
        ce1, ce2, ce_whole = estimate_cond_entropy(s1), estimate_cond_entropy(s2), estimate_cond_entropy(s_whole)
        feats.update({
            'cond_entropy_left': ce1, 'cond_entropy_right': ce2, 'cond_entropy_whole': ce_whole,
            'cond_entropy_diff': ce2 - ce1, 'cond_entropy_ratio': ce2 / (ce1 + 1e-6)
        })
        _add_contribution_ratio_feats(feats, 'cond_entropy', ce1, ce2, ce_whole)
    except Exception:
        feats.update({'cond_entropy_left':0, 'cond_entropy_right':0, 'cond_entropy_whole':0, 'cond_entropy_diff':0, 'cond_entropy_ratio':0})
    
    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 9. 分形 ---
@register_feature(func_id="9")
def fractal_dimension_features(u: pd.DataFrame) -> dict:
    s1 = u['value'][u['period'] == 0].to_numpy()
    s2 = u['value'][u['period'] == 1].to_numpy()
    s_whole = u['value'].to_numpy()
    feats = {}
    
    fractal_funcs = {
        'petrosian_fd': antropy.petrosian_fd,
        'katz_fd': antropy.katz_fd,
        'higuchi_fd': antropy.higuchi_fd,
        'detrended_fluctuation': antropy.detrended_fluctuation,
    }

    for name, func in fractal_funcs.items():
        try:
            v1, v2, v_whole = func(s1), func(s2), func(s_whole)
            feats[f'{name}_left'] = v1
            feats[f'{name}_right'] = v2
            feats[f'{name}_whole'] = v_whole
            feats[f'{name}_diff'] = v2 - v1
            feats[f'{name}_ratio'] = v2 / (v1 + 1e-6)
        except Exception:
            feats.update({f'{name}_left': 0, f'{name}_right': 0, f'{name}_whole': 0, f'{name}_diff': 0, f'{name}_ratio': 0})

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 10. tsfresh --- 
@register_feature(func_id="10")
def tsfresh_features(u: pd.DataFrame) -> dict:
    """基于tsfresh的特征工程"""
    s1 = u['value'][u['period'] == 0].to_numpy()
    s2 = u['value'][u['period'] == 1].to_numpy()
    s_whole = u['value'].to_numpy()
    feats = {}

    funcs = {
        tsfresh_fe.ratio_value_number_to_time_series_length: None,
        tsfresh_fe.ratio_beyond_r_sigma: [6, 1.5],
        tsfresh_fe.quantile: [0.6, 0.4, 0.1],
        tsfresh_fe.percentage_of_reoccurring_values_to_all_values: None,
        tsfresh_fe.percentage_of_reoccurring_datapoints_to_all_datapoints: None,
        tsfresh_fe.last_location_of_maximum: None,
        tsfresh_fe.first_location_of_maximum: None,
        tsfresh_fe.partial_autocorrelation: [{"lag": 2}],
        tsfresh_fe.linear_trend: [{"attr": "slope"}, {"attr": "rvalue"}, {"attr": "intercept"}],
        tsfresh_fe.fft_coefficient: [{"coeff": 3, "attr": "imag"}, {"coeff": 2, "attr": "imag"}, {"coeff": 1, "attr": "imag"}],
        tsfresh_fe.change_quantiles: [
            {"f_agg": "var", "isabs": True,  "qh": 1.0, "ql": 0.4},
            {"f_agg": "var", "isabs": True,  "qh": 1.0, "ql": 0.2},
            {"f_agg": "var", "isabs": True,  "qh": 0.8, "ql": 0.6},
            {"f_agg": "var", "isabs": True,  "qh": 0.8, "ql": 0.4},
            {"f_agg": "var", "isabs": True,  "qh": 0.8, "ql": 0.2},
            {"f_agg": "var", "isabs": True,  "qh": 0.6, "ql": 0.4},
            {"f_agg": "var", "isabs": True,  "qh": 0.6, "ql": 0.2},
            {"f_agg": "var", "isabs": True,  "qh": 0.4, "ql": 0.2},
            {"f_agg": "var", "isabs": False, "qh": 1.0, "ql": 0.4},
            {"f_agg": "var", "isabs": False, "qh": 1.0, "ql": 0.2},
            {"f_agg": "var", "isabs": False, "qh": 0.8, "ql": 0.4},
            {"f_agg": "var", "isabs": False, "qh": 0.8, "ql": 0.2},
            {"f_agg": "var", "isabs": False, "qh": 0.8, "ql": 0.0},
            {"f_agg": "var", "isabs": False, "qh": 0.6, "ql": 0.4},
            {"f_agg": "var", "isabs": False, "qh": 0.6, "ql": 0.2},
            {"f_agg": "var", "isabs": False, "qh": 0.4, "ql": 0.2},
            {"f_agg": "mean","isabs": True,  "qh": 1.0, "ql": 0.4},
            {"f_agg": "mean","isabs": True,  "qh": 0.6, "ql": 0.4},
        ],
        tsfresh_fe.ar_coefficient: [{"coeff": 2, "k": 10}],
        tsfresh_fe.agg_linear_trend: [
            {"attr": "slope", "chunk_len": 50, "f_agg": "mean"},
            {"attr": "slope", "chunk_len": 5,  "f_agg": "mean"},
            {"attr": "slope", "chunk_len": 10, "f_agg": "mean"},
            {"attr": "rvalue", "chunk_len": 50, "f_agg": "mean"},
            {"attr": "rvalue", "chunk_len": 50, "f_agg": "max"},
            {"attr": "rvalue", "chunk_len": 5,  "f_agg": "mean"},
            {"attr": "rvalue", "chunk_len": 5,  "f_agg": "max"},
            {"attr": "rvalue", "chunk_len": 10, "f_agg": "mean"},
            {"attr": "rvalue", "chunk_len": 10, "f_agg": "max"},
            {"attr": "intercept", "chunk_len": 50, "f_agg": "mean"},
            {"attr": "intercept", "chunk_len": 50, "f_agg": "max"},
            {"attr": "intercept", "chunk_len": 5,  "f_agg": "mean"},
            {"attr": "intercept", "chunk_len": 5,  "f_agg": "max"},
            {"attr": "intercept", "chunk_len": 10, "f_agg": "mean"},
            {"attr": "intercept", "chunk_len": 10, "f_agg": "max"},
        ]
    }

    def param_to_str(param):
        if isinstance(param, dict):
            return '_'.join([f"{k}_{v}" for k, v in param.items()])
        else:
            return str(param)

    def calculate_stats_for_feature(func, param=None):
        results = {}
        base_name = func.__name__
        if param is not None:
            base_name += f"_{param_to_str(param)}"

        try:
            # Prepare arguments for each segment
            args_s1 = [s1]
            args_s2 = [s2]
            args_s_whole = [s_whole]
            is_combiner = False

            if param is None: # Simple function, no params
                pass
            elif isinstance(param, dict):
                # Check if it's a combiner function or a function with kwargs
                sig = inspect.signature(func)
                if 'param' in sig.parameters: # Combiner function
                    is_combiner = True
                    args_s1.append([param])
                    args_s2.append([param])
                    args_s_whole.append([param])
                else: # Function with kwargs
                    args_s1.append(param)
                    args_s2.append(param)
                    args_s_whole.append(param)
            else: # Simple function with a single parameter
                args_s1.append(param)
                args_s2.append(param)
                args_s_whole.append(param)

            # Execute function for each segment
            if is_combiner:
                v1_dict = {k: v for k, v in func(*args_s1)}
                v2_dict = {k: v for k, v in func(*args_s2)}
                v_whole_dict = {k: v for k, v in func(*args_s_whole)}
                
                for key in v1_dict:
                    v1, v2, v_whole = v1_dict[key], v2_dict[key], v_whole_dict[key]
                    feat_name_base = f"{func.__name__}_{key}"
                    results[f'{feat_name_base}_left'] = v1
                    results[f'{feat_name_base}_right'] = v2
                    results[f'{feat_name_base}_whole'] = v_whole
                    results[f'{feat_name_base}_diff'] = v2 - v1
                    results[f'{feat_name_base}_ratio'] = v2 / (v1 + 1e-6)
                    _add_contribution_ratio_feats(results, feat_name_base, v1, v2, v_whole)
                return results

            else:
                if isinstance(param, dict) and not is_combiner:
                    v1, v2, v_whole = func(args_s1[0], **args_s1[1]), func(args_s2[0], **args_s2[1]), func(args_s_whole[0], **args_s_whole[1])
                else:
                    v1, v2, v_whole = func(*args_s1), func(*args_s2), func(*args_s_whole)

                results[f'{base_name}_left'] = v1
                results[f'{base_name}_right'] = v2
                results[f'{base_name}_whole'] = v_whole
                results[f'{base_name}_diff'] = v2 - v1
                results[f'{base_name}_ratio'] = v2 / (v1 + 1e-6)
                _add_contribution_ratio_feats(results, base_name, v1, v2, v_whole)
        
        except Exception:
            # For combiner functions, need to know keys to create nulls
            if 'param' in locals() and inspect.isfunction(func) and 'param' in inspect.signature(func).parameters:
                 # It's a combiner, but we can't get keys without running it. Skip for now on error.
                 pass
            else:
                results[f'{base_name}_left'] = np.nan
                results[f'{base_name}_right'] = np.nan
                results[f'{base_name}_whole'] = np.nan
                results[f'{base_name}_diff'] = np.nan
                results[f'{base_name}_ratio'] = np.nan
                
        return results


    for func, params in funcs.items():
        if params is None:
            feats.update(calculate_stats_for_feature(func))
        else:
            for param in params:
                feats.update(calculate_stats_for_feature(func, param))

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 11. 时间序列建模 ---
@register_feature(func_id="11")
def ar_model_features(u: pd.DataFrame) -> dict:
    """
    基于AR模型派生特征。
    1. 在 period 0 上训练模型，预测 period 1，计算残差统计量。
    2. 在 period 1 上训练模型，预测 period 0，计算残差统计量。
    3. 分别在 period 0 和 1 上训练模型，比较模型参数、残差和信息准则(AIC/BIC)。
    """
    s1 = u['value'][u['period'] == 0].to_numpy()
    s2 = u['value'][u['period'] == 1].to_numpy()
    s_whole = u['value'].to_numpy()
    feats = {}
    lags = 5 # 固定阶数以保证可比性

    # --- 特征组1: 用 s1 训练，预测 s2 ---
    if len(s1) > lags and len(s2) > 0:
        try:
            model1_fit = AutoReg(s1, lags=lags).fit()
            predictions = model1_fit.predict(start=len(s1), end=len(s1) + len(s2) - 1, dynamic=True)
            residuals = s2 - predictions
            feats['ar_residuals_s2_pred_mean'] = np.mean(residuals)
            feats['ar_residuals_s2_pred_std'] = np.std(residuals)
            feats['ar_residuals_s2_pred_skew'] = pd.Series(residuals).skew()
            feats['ar_residuals_s2_pred_kurt'] = pd.Series(residuals).kurt()
        except Exception:
            # 宽泛地捕获异常，防止因数值问题中断
            feats.update({'ar_residuals_s2_pred_mean': 0, 'ar_residuals_s2_pred_std': 0, 'ar_residuals_s2_pred_skew': 0, 'ar_residuals_s2_pred_kurt': 0})
    else:
        feats.update({'ar_residuals_s2_pred_mean': 0, 'ar_residuals_s2_pred_std': 0, 'ar_residuals_s2_pred_skew': 0, 'ar_residuals_s2_pred_kurt': 0})

    # --- 特征组2: 用 s2 训练，预测 s1 ---
    if len(s2) > lags and len(s1) > 0:
        try:
            model2_fit = AutoReg(s2, lags=lags).fit()
            predictions_on_s1 = model2_fit.predict(start=len(s2), end=len(s2) + len(s1) - 1, dynamic=True)
            residuals_s1_pred = s1 - predictions_on_s1
            feats['ar_residuals_s1_pred_mean'] = np.mean(residuals_s1_pred)
            feats['ar_residuals_s1_pred_std'] = np.std(residuals_s1_pred)
            feats['ar_residuals_s1_pred_skew'] = pd.Series(residuals_s1_pred).skew()
            feats['ar_residuals_s1_pred_kurt'] = pd.Series(residuals_s1_pred).kurt()
        except Exception:
            feats.update({'ar_residuals_s1_pred_mean': 0, 'ar_residuals_s1_pred_std': 0, 'ar_residuals_s1_pred_skew': 0, 'ar_residuals_s1_pred_kurt': 0})
    else:
        feats.update({'ar_residuals_s1_pred_mean': 0, 'ar_residuals_s1_pred_std': 0, 'ar_residuals_s1_pred_skew': 0, 'ar_residuals_s1_pred_kurt': 0})


    # --- 特征组3: 分别建模，比较差异 ---
    s1_resid_std, s1_params = np.nan, np.full(lags + 1, np.nan)
    s1_aic, s1_bic = np.nan, np.nan
    if len(s1) > lags:
        try:
            fit1 = AutoReg(s1, lags=lags).fit()
            s1_resid_std = np.std(fit1.resid)
            s1_params = fit1.params
            s1_aic = fit1.aic
            s1_bic = fit1.bic
        except Exception:
            pass

    s2_resid_std, s2_params = np.nan, np.full(lags + 1, np.nan)
    s2_aic, s2_bic = np.nan, np.nan
    if len(s2) > lags:
        try:
            fit2 = AutoReg(s2, lags=lags).fit()
            s2_resid_std = np.std(fit2.resid)
            s2_params = fit2.params
            s2_aic = fit2.aic
            s2_bic = fit2.bic
        except Exception:
            pass

    swhole_resid_std, swhole_params = np.nan, np.full(lags + 1, np.nan)
    swhole_aic, swhole_bic = np.nan, np.nan
    if len(s_whole) > lags:
        try:
            fit_whole = AutoReg(s_whole, lags=lags).fit()
            swhole_resid_std = np.std(fit_whole.resid)
            swhole_params = fit_whole.params
            swhole_aic = fit_whole.aic
            swhole_bic = fit_whole.bic
        except Exception:
            pass
            
    feats['ar_resid_std_left'] = s1_resid_std
    feats['ar_resid_std_right'] = s2_resid_std
    feats['ar_resid_std_whole'] = swhole_resid_std
    feats['ar_resid_std_diff'] = (s2_resid_std - s1_resid_std) if not (np.isnan(s1_resid_std) or np.isnan(s2_resid_std)) else 0
    feats['ar_resid_std_ratio'] = (s2_resid_std / (s1_resid_std + 1e-6)) if not (np.isnan(s1_resid_std) or np.isnan(s2_resid_std)) else 0
    _add_contribution_ratio_feats(feats, 'ar_resid_std', s1_resid_std, s2_resid_std, swhole_resid_std)
    
    feats['ar_aic_left'] = s1_aic
    feats['ar_aic_right'] = s2_aic
    feats['ar_aic_whole'] = swhole_aic
    feats['ar_aic_diff'] = (s2_aic - s1_aic) if not (np.isnan(s1_aic) or np.isnan(s2_aic)) else 0
    feats['ar_aic_ratio'] = (s2_aic / (s1_aic + 1e-6)) if not (np.isnan(s1_aic) or np.isnan(s2_aic)) else 0
    _add_contribution_ratio_feats(feats, 'ar_aic', s1_aic, s2_aic, swhole_aic)

    feats['ar_bic_left'] = s1_bic
    feats['ar_bic_right'] = s2_bic
    feats['ar_bic_whole'] = swhole_bic
    feats['ar_bic_diff'] = (s2_bic - s1_bic) if not (np.isnan(s1_bic) or np.isnan(s2_bic)) else 0
    feats['ar_bic_ratio'] = (s2_bic / (s1_bic + 1e-6)) if not (np.isnan(s1_bic) or np.isnan(s2_bic)) else 0
    _add_contribution_ratio_feats(feats, 'ar_bic', s1_bic, s2_bic, swhole_bic)
    
    # 比较模型系数
    for i in range(len(s1_params)):
        feats[f'param_{i}_left'] = s1_params[i]
        feats[f'param_{i}_right'] = s2_params[i]
        feats[f'param_{i}_whole'] = swhole_params[i]
        feats[f'param_{i}_diff'] = s2_params[i] - s1_params[i]
        feats[f'param_{i}_ratio'] = s2_params[i] / (s1_params[i] + 1e-6)
        _add_contribution_ratio_feats(feats, f'param_{i}', s1_params[i], s2_params[i], swhole_params[i])

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 12. 分段损失 ---
class RPTFeatureExtractor:
    def __init__(self):
        # 所有可用的cost类及其名称
        self.cost_classes = {
            'l1': rpt.costs.CostL1,               # 中位数
            'l2': rpt.costs.CostL2,               # 均值
            'clinear': rpt.costs.CostCLinear,     # 线性协方差
            'rbf': rpt.costs.CostRbf,             # RBF核
            'normal': rpt.costs.CostNormal,       # 协方差
            'ar': rpt.costs.CostAR,               # 自回归
            'mahalanobis': rpt.costs.CostMl,      # 马氏距离
            'rank': rpt.costs.CostRank,           # 排名
            'cosine': rpt.costs.CostCosine,       # 余弦距离
        }

    def calculate(self, cost, start, end):
        result = cost.error(start, end)
        if isinstance(result, (np.ndarray, list)) and np.array(result).size == 1:
            return float(np.array(result).squeeze())
        return result

    def extract(self, signal, boundary):
        """
        输入：
            signal: 1D numpy array，单变量时间序列
            boundary: int，分割点
        输出：
            result: dict，格式为 {cost_name: {'left': value, 'right': value}}
        """
        signal = np.asarray(signal)
        n = len(signal)
        result = {}
        for name, cls in self.cost_classes.items():
            try:
                if name == 'ar':
                    cost = cls(order=4)
                else:
                    cost = cls()
                cost.fit(signal)
                left = self.calculate(cost, 0, boundary)
                right = self.calculate(cost, boundary, n)
                whole = self.calculate(cost, 0, n)
                diff = right - left if left is not None and right is not None else None
                ratio = right / (left + 1e-6) if left is not None and right is not None else None
            except Exception:
                left = None
                right = None
                whole = None
                diff = None
                ratio = None
            result[name] = {'left': left, 'right': right, 'whole': whole, 'diff': diff, 'ratio': ratio}
        return result

@register_feature(func_id="12")
def rupture_cost_features(u: pd.DataFrame) -> dict:
    value = u['value'].values.astype(np.float32)
    period = u['period'].values.astype(np.float32)
    boundary = np.where(np.diff(period) != 0)[0].item()
    feats = {}

    extractor = RPTFeatureExtractor()
    features = extractor.extract(value, boundary)

    feats = {}
    for k, v in features.items():
        for seg, value in v.items():
            feats[f'rpt_cost_{k}_{seg}'] = value
        _add_contribution_ratio_feats(feats, f'rpt_cost_{k}', v['left'], v['right'], v['whole'])

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 时序变换函数注册表 ---
TRANSFORM_REGISTRY = {}

def register_transform(_func=None, *, output_mode_names=[]):
    """一个用于注册时序变换函数的装饰器。"""
    def decorator_register(func):
        TRANSFORM_REGISTRY[func.__name__] = {
            "func": func, 
            "output_mode_names": output_mode_names
        }
        return func

    if _func is None:
        # Used as @register_transform(output_mode_names=...)
        return decorator_register
    else:
        # Used as @register_transform
        return decorator_register(_func)

@register_transform(output_mode_names=['RAW'])
def no_transformation(X_df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    原始时序
    """
    result_dfs = []
    result_dfs.append(X_df)

    return result_dfs

# @register_transform(output_mode_names=['MAde_trend', 'MAde_resid'])
def moving_average_decomposition(X_df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    滑动平均分解
    
    Args:
        X_df: 输入数据框，包含MultiIndex (id, time) 和 columns ['value', 'period']
        
    Returns:
        List[pd.DataFrame]: 包含两个数据框的列表 [趋势值, 残差值]
    """
    X_df_sorted = X_df.sort_index()
    result_dfs = []
    
    # 为每个模态创建一个空的数据框
    for mode_name in ['trend', 'resid']:
        mode_df = X_df_sorted.copy()
        mode_df['value'] = np.nan
        result_dfs.append(mode_df)
    
    # 对每个id进行分解
    for series_id in X_df_sorted.index.get_level_values('id').unique():
        series_data = X_df_sorted.loc[series_id]
        series_data = series_data.sort_index()
        values = series_data['value'].values
        
        # 滑动平均分解
        window_size = 200
        trend = pd.Series(values).rolling(window=window_size, center=True, min_periods=1).mean()
        trend.iloc[:window_size//2] = trend.iloc[window_size//2]
        trend.iloc[-(window_size//2):] = trend.iloc[-(window_size//2)]
        
        residual = values - trend.values
        
        result_dfs[0].loc[series_id, 'value'] = trend.values  # 趋势值
        result_dfs[1].loc[series_id, 'value'] = residual  # 残差值
    
    return result_dfs

# --- 特征管理核心逻辑 ---
def _get_latest_feature_file() -> Path | None:
    """查找并返回最新的特征文件路径"""
    # 获取特征文件目录下的所有特征文件
    feature_files = list(config.FEATURE_DIR.glob('features_*.parquet'))
    # 如果没有特征文件，返回None
    if not feature_files:
        return None
    return max(feature_files, key=lambda p: p.stat().st_mtime)

def _load_feature_file(file_path: Path):
    """加载指定的特征文件及其元数据。"""
    if not file_path or not file_path.exists():
        return pd.DataFrame(), {}
    try:
        table = pd.read_parquet(file_path)
        metadata_str = table.attrs.get('feature_metadata', '{}')
        metadata = json.loads(metadata_str)
        return table, metadata
    except Exception as e:
        logger.warning(f"无法加载特征文件 {file_path}: {e}。")
        return pd.DataFrame(), {}

def _save_feature_file(df: pd.DataFrame, metadata: dict) -> Path:
    """将特征保存到一个新的、带时间戳的文件中，并返回其路径。"""
    config.FEATURE_DIR.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_feature_path = config.FEATURE_DIR / f'features_{timestamp}.parquet'
    
    df.attrs['feature_metadata'] = json.dumps(metadata)
    df.to_parquet(new_feature_path)
    return new_feature_path

def _backup_feature_file(file_path: Path):
    """如果文件路径有效，则备份它。"""
    if file_path and file_path.exists():
        config.FEATURE_BACKUP_DIR.mkdir(exist_ok=True, parents=True)
        backup_path = config.FEATURE_BACKUP_DIR / file_path.name
        file_path.rename(backup_path)
        logger.info(f"已将文件 {file_path.name} 备份到: {config.FEATURE_BACKUP_DIR}")

def _apply_feature_func_sequential(func, X_df: pd.DataFrame) -> pd.DataFrame:
    """顺序应用单个特征函数"""
    all_ids = X_df.index.get_level_values("id").unique()
    results = [
        {**{'id': id_val}, **func(X_df.loc[id_val])}
        for id_val in tqdm(all_ids, desc=f"Running {func.__name__} (sequentially)")
    ]
    return pd.DataFrame(results).set_index('id')

def _apply_feature_func_parallel(func, X_df: pd.DataFrame) -> pd.DataFrame:
    """并行应用单个特征函数"""
    all_ids = X_df.index.get_level_values("id").unique()
    results = Parallel(n_jobs=64)(
        delayed(lambda df_id, id_val: {**{'id': id_val}, **func(df_id)})(X_df.loc[id_val], id_val)
        for id_val in tqdm(all_ids, desc=f"Running {func.__name__}")
    )
    return pd.DataFrame(results).set_index('id')

def _apply_transform_func(func, X_df: pd.DataFrame) -> List[pd.DataFrame]:
    """执行变换函数"""
    return func(X_df)

def apply_transformation(X_df: pd.DataFrame, transform_funcs: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    应用时序变换
    
    Args:
        X_df: 输入数据框
        transform_funcs: 要应用的变换函数名称列表，如果为None则应用所有注册的变换函数
        
    Returns:
        Dict[str, pd.DataFrame]: 键为模态名称，值为对应的数据框
    """
    if transform_funcs is None:
        transform_funcs = list(TRANSFORM_REGISTRY.keys())
    
    # 验证变换函数是否存在
    valid_transform_funcs = []
    for func_name in transform_funcs:
        if func_name not in TRANSFORM_REGISTRY:
            logger.warning(f"变换函数 {func_name} 未在注册表中找到，已跳过。")
        else:
            valid_transform_funcs.append(func_name)
    
    transform_funcs = valid_transform_funcs
    
    # 存储所有模态的数据框
    transformed_data = {}
    
    for func_name in transform_funcs:
        logger.info(f"--- 开始应用变换函数: {func_name} ---")
        start_time = time.time()
        
        transform_info = TRANSFORM_REGISTRY[func_name]
        func = transform_info['func']
        output_mode_names = transform_info['output_mode_names']
        
        # 执行变换
        transformed_results = _apply_transform_func(func, X_df)
        
        # 存储结果
        for mode_name, mode_df in zip(output_mode_names, transformed_results):
            transformed_data[mode_name] = mode_df
        
        duration = time.time() - start_time
        logger.info(f"'{func_name}' 变换完毕，耗时: {duration:.2f} 秒，生成模态: {output_mode_names}")
    
    return transformed_data

def check_new_features_corr(feature_df, loaded_features, drop_flag=False, threshold=0.95):
    """检查新特征与已加载特征的相关性"""
    new_features = [col for col in feature_df.columns if col not in loaded_features]
    logger.info(f"\nNumber of new features: {len(new_features)}")
    logger.info(f"Number of loaded features: {len(loaded_features)}")
    
    # 计算新特征与已加载特征的相关性
    corr_matrix = feature_df[new_features + list(loaded_features)].corr()
    cross_corr = corr_matrix.loc[new_features, loaded_features]
    high_corr_features = cross_corr[(cross_corr.abs() > 0.7).any(axis=1)]
    
    if not high_corr_features.empty:
        logger.info("\nNew features with high correlation (|corr| > 0.7) to loaded features:")
        # 打印每个高相关性新特征及其相关特征
        for new_feat in high_corr_features.index:
            correlated_with = high_corr_features.columns[high_corr_features.loc[new_feat].abs() > 0.7]
            corr_values = high_corr_features.loc[new_feat, high_corr_features.loc[new_feat].abs() > 0.7]
            
            logger.info(f"\n{new_feat} is highly correlated with:")
            for loaded_feat, corr in zip(correlated_with, corr_values):
                logger.info(f"  - {loaded_feat}: {corr:.3f}")
    else:
        logger.info("\nNo new features show high correlation (|corr| > 0.7) with loaded features.")
        
    # 删除高度相关的新特征（严格大于 threshold）
    dropped_features = []
    if drop_flag:
        high_corr_to_drop = cross_corr[(cross_corr.abs() > threshold).any(axis=1)]
        dropped_features = list(high_corr_to_drop.index)
        if dropped_features:
            logger.info(f"\nDropping {len(dropped_features)} new features with |corr| > {threshold}:")
            for feat in dropped_features:
                logger.info(f"  - {feat}")
            feature_df = feature_df.drop(columns=dropped_features)
        else:
            logger.info(f"\nNo new features exceeded threshold |corr| > {threshold}, nothing dropped.")

    return feature_df, dropped_features

def clean_feature_names(df: pd.DataFrame, prefix: str = "f") -> pd.DataFrame:
    """清理特征名称，确保它们是合法的列名。"""
    cleaned_columns = []
    for i, col in enumerate(df.columns):
        # 替换非法字符为 _
        cleaned = re.sub(r'[^\w]', '_', col)
        # 防止开头是数字（如 "123_feature"）非法
        if re.match(r'^\d', cleaned):
            cleaned = f"{prefix}_{cleaned}"
        # 多个连续 _ 合并为一个
        cleaned = re.sub(r'__+', '_', cleaned)
        cleaned_columns.append(cleaned)
    df.columns = cleaned_columns
    return df

def generate_features(
        X_df: pd.DataFrame, 
        funcs_to_run: list = None, 
        trans_to_run: list = None, 
        base_feature_file: str = None
    ):
    """
    生成指定的特征，或者如果未指定，则生成所有已注册的特征。
    可以基于一个现有的特征文件进行增量更新。

    Args:
        X_df (pd.DataFrame): 包含 `series_id` 和 `time_step` 的输入数据。
        funcs_to_run (list, optional): 要运行的特征函数名称列表。
            如果为 None，则运行所有在 `FEATURE_REGISTRY` 中注册的、且不在 `EXPERIMENTAL_FEATURES` 中的函数。
        trans_to_run (list, optional): 要运行的变换函数名称列表。
        base_feature_file (str, optional): 基础特征文件名。如果提供，
            将加载此文件并在此基础上添加或更新特征。否则，将创建一个新的特征集。
        clip_outliers (bool): 是否在特征工程前裁剪极端离群值。默认为 True。
        clip_threshold (float): 定义离群值的IQR乘数。默认为 5.0。
    """
    utils.ensure_feature_dirs()
    
    if funcs_to_run is None:
        # 如果未指定函数，则运行所有非实验性特征
        funcs_to_run = [
            f for f in FEATURE_REGISTRY.keys() 
            if f not in config.EXPERIMENTAL_FEATURES
        ]
        logger.info(f"未指定特征函数，将运行所有 {len(funcs_to_run)} 个非实验性特征。")
    
    # 验证请求的函数是否都已注册
    valid_funcs_to_run = []
    for func_name in funcs_to_run:
        if func_name not in FEATURE_REGISTRY:
            logger.warning(f"函数 {func_name} 未在注册表中找到，已跳过。")
        else:
            valid_funcs_to_run.append(func_name)
    
    funcs_to_run = valid_funcs_to_run

    # 1. 确定基础特征文件
    if base_feature_file:
        base_path = config.FEATURE_DIR / base_feature_file
    else:
        base_path = _get_latest_feature_file()

    # 2. 加载基础特征
    if base_path and base_path.exists():
        logger.info(f"将基于特征文件进行更新: {base_path.name}")
        feature_df, metadata = _load_feature_file(base_path)
        # 更新操作需要备份旧文件
        # _backup_feature_file(base_path) # <<< 移除此处的调用
    else:
        logger.info("未找到基础特征文件，将创建全新的特征集。")
        feature_df, metadata = pd.DataFrame(index=X_df.index.get_level_values('id').unique()), {}
    logger.info(f"基础特征文件格式：{feature_df.shape}")

    # 3. 时序分解
    logger.info("=== 开始时序分解 ===")
    transformed_data = apply_transformation(X_df, trans_to_run)
    logger.info(f"分解完成，共生成 {len(transformed_data)} 个模态: {list(transformed_data.keys())}")

    # 4. 逐个生成新特征并更新
    loaded_features = feature_df.columns.tolist()
    initial_feature_count = len(feature_df.columns)

    for mode_name, mode_df in transformed_data.items():
        logger.info(f"=== 开始为模态 '{mode_name}' 生成特征 ===")
        for func_name in funcs_to_run:
            logger.info(f"--- 开始生成特征: {func_name} ---")
            start_time = time.time()
            
            feature_info = FEATURE_REGISTRY[func_name]
            func = feature_info['func']
            is_parallelizable = feature_info['parallelizable']
            func_id = feature_info['func_id']
            
            if is_parallelizable:
                new_features_df = _apply_feature_func_parallel(func, mode_df)
            else:
                logger.info(f"函数 '{func_name}' 不可并行化，将顺序执行。")
                new_features_df = _apply_feature_func_sequential(func, mode_df)
            new_features_df.columns = [f"{mode_name}_{func_id}_{col}" for col in new_features_df.columns]

            # 记录日志
            duration = time.time() - start_time
            logger.info(f"'{func_name}' 生成完毕，耗时: {duration:.2f} 秒。")
            logger.info(f"  新生成特征列名: {new_features_df.columns.tolist()}")
            
            for col in new_features_df.columns:
                null_ratio = new_features_df[col].isnull().sum() / len(new_features_df)
                zero_ratio = (new_features_df[col] == 0).sum() / len(new_features_df)
                logger.info(f"    - '{col}': 空值比例={null_ratio:.2%}, 零值比例={zero_ratio:.2%}")

            # 删除旧版本特征（如果存在），然后合并
            feature_df = feature_df.drop(columns=new_features_df.columns, errors='ignore')
            feature_df = feature_df.merge(new_features_df, left_index=True, right_index=True, how='left')
            feature_df, removed_features = check_new_features_corr(feature_df, loaded_features, drop_flag=True, threshold=0.95)
            feature_df = clean_feature_names(feature_df)
            loaded_features = feature_df.columns.tolist()

    # 3. 保存结果
    new_feature_count = len(feature_df.columns)
    if new_feature_count > initial_feature_count:
        # 只有在成功生成新特征后，才备份旧文件
        if base_path and base_path.exists():
            _backup_feature_file(base_path)
            
        metadata['last_updated_funcs'] = funcs_to_run
        new_file_path = _save_feature_file(feature_df, metadata)
        logger.info(f"特征已保存到新文件: {new_file_path.name}")
        logger.info("--- 生成后完整特征列表 ---")
        logger.info(f"{feature_df.columns.tolist()}")
        logger.info("-----------------------------")
    else:
        logger.info("没有新特征生成，文件未保存。")

    final_feature_count = len(feature_df.columns)
    logger.info(f"生成/更新完成。新文件: {new_file_path.name if new_feature_count > initial_feature_count else 'N/A'}, 总特征数: {final_feature_count}")

def delete_features(base_feature_file: str, funcs_to_delete: list = None, cols_to_delete: list = None):
    """
    从指定的特征文件中删除特征，并生成一个新的带时间戳的文件。
    可以按函数名删除，也可以按特定的列名删除。
    """
    base_path = config.FEATURE_DIR / base_feature_file
    if not base_path.exists():
        logger.error(f"指定的特征文件不存在: {base_feature_file}")
        return

    logger.info(f"将从文件 {base_feature_file} 中删除特征...")
    feature_df, metadata = _load_feature_file(base_path)
    
    if feature_df.empty:
        logger.error(f"无法从 {base_feature_file} 加载数据，操作中止。")
        return

    _backup_feature_file(base_path)
    
    initial_columns = set(feature_df.columns)
    cols_to_drop = []

    if funcs_to_delete:
        logger.info(f"按函数名删除: {funcs_to_delete}")
        # 修复逻辑: 检查 metadata['last_updated_funcs'] 中生成的列
        # 旧逻辑依赖于 metadata[func_name]，但这个key没有被创建
        generated_cols_by_func = {}
        # 假设 metadata['generated_by'][col] = func_name 的结构，需要先改造 generate_features
        # 当前元数据不健全，无法安全地按函数名删除。
        # 作为一个临时的、更鲁棒的方案，我们基于函数名约定来猜测列名
        for func_name in funcs_to_delete:
            # 这是一个简化的匹配，可能会误删。更可靠的方式是改造 generate_features 来记录每个特征由谁生成。
            # 例如，'distributional_stats' 会匹配 'distributional_stats_mean_diff' 等
            # 为了安全，我们只匹配以函数名开头的列
            matched_cols = [col for col in feature_df.columns if col.startswith(func_name)]
            if matched_cols:
                logger.info(f"  函数 '{func_name}' 匹配到 {len(matched_cols)} 列: {matched_cols}")
                cols_to_drop.extend(matched_cols)
            else:
                logger.warning(f"  未找到与函数 '{func_name}' 相关的特征列。")


    if cols_to_delete:
        logger.info(f"按列名删除: {cols_to_delete}")
        # 验证列名是否存在
        valid_cols = [c for c in cols_to_delete if c in feature_df.columns]
        invalid_cols = set(cols_to_delete) - set(valid_cols)
        if invalid_cols:
            logger.warning(f"  以下列名不存在，将被忽略: {list(invalid_cols)}")
        cols_to_drop.extend(valid_cols)

    # 去重并执行删除
    final_cols_to_drop = sorted(list(set(cols_to_drop)))
    if not final_cols_to_drop:
        logger.warning("没有找到任何要删除的特征列，操作中止。")
        # 恢复备份，因为没有变化
        backup_path = config.FEATURE_BACKUP_DIR / base_path.name
        if backup_path.exists():
            backup_path.rename(base_path)
            logger.info("已恢复原始文件。")
        return
        
    logger.info(f"总计将删除 {len(final_cols_to_drop)} 个特征列: {final_cols_to_drop}")
    feature_df.drop(columns=final_cols_to_drop, inplace=True)
    
    # 更新元数据（简单地移除一个标记，表示文件被修改过）
    metadata['last_deleted_features'] = final_cols_to_drop
    
    new_path = _save_feature_file(feature_df, metadata)
    logger.info(f"删除完成。新文件: {new_path.name}, 总特征数: {len(feature_df.columns)}")


def load_features(feature_file: str = None) -> tuple[pd.DataFrame | None, str | None]:
    """加载指定的或最新的特征文件。"""
    if feature_file:
        path_to_load = config.FEATURE_DIR / feature_file
    else:
        logger.info("未指定特征文件，将尝试加载最新版本。")
        path_to_load = _get_latest_feature_file()

    if not path_to_load or not path_to_load.exists():
        logger.error(f"无法找到要加载的特征文件: {path_to_load}")
        return None, None

    logger.info(f"正在从 {path_to_load.name} 加载特征...")
    feature_df, _ = _load_feature_file(path_to_load)
    
    if feature_df.empty:
        return None, None
        
    logger.info(f"特征加载成功，共 {len(feature_df.columns)} 个特征。")
    return feature_df, path_to_load.name 
