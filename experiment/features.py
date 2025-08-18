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

def _add_diff_ratio_feats(feats: dict, name: str, left, right):
    """
    一个辅助函数，用于向特征字典中添加差异和比例特征。

    Args:
        feats (dict): 要更新的特征字典。
        name (str): 特征的基础名称 (例如, 'stats_mean')。
        left (float): 左侧分段的特征值。
        right (float): 右侧分段的特征值。
    """
    # check nan/None 
    if np.isnan(left) or np.isnan(right) or left is None or right is None:
        feats[f'{name}_diff'] = 0.0
        feats[f'{name}_ratio'] = 0.0
        return
    # 做差
    feats[f'{name}_diff'] = right - left
    # 做比
    feats[f'{name}_ratio'] = right / (left + 1e-6)


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
    # check nan/None 
    if np.isnan(left) or np.isnan(right) or np.isnan(whole) or left is None or right is None or whole is None :
        feats[f'{name}_contribution_left'] = 0.0
        feats[f'{name}_contribution_right'] = 0.0
        feats[f'{name}_ratio_to_whole_left'] = 0.0
        feats[f'{name}_ratio_to_whole_right'] = 0.0
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

def rolling_std_mean(s, window=50):
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
                # diff = right - left
                # ratio = right / (left + 1e-6)
            except Exception:
                left = None
                right = None
                whole = None
                # diff = None
                # ratio = None
            # Move to _add_diff_ratio_feats, 'diff': diff, 'ratio': ratio
            result[name] = {'left': left, 'right': right, 'whole': whole}   
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
        _add_diff_ratio_feats(feats, f'stats_{k}', v['left'], v['right'])
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
    w_stat, w_pvalue = scipy.stats.ranksums(s1, s2)
    feats['wilcoxon_stat'] = w_stat if not np.isnan(w_stat) else 0
    feats['wilcoxon_pvalue'] = -w_pvalue if not np.isnan(w_pvalue) else 1

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
    try:
        sw1_stat, sw1_pvalue = scipy.stats.shapiro(s1)
        sw2_stat, sw2_pvalue = scipy.stats.shapiro(s2)
        sw_whole_stat, sw_whole_pvalue = scipy.stats.shapiro(s_whole)
    except:
        pass
    feats['shapiro_pvalue_left'] = sw1_pvalue
    feats['shapiro_pvalue_right'] = sw2_pvalue
    feats['shapiro_pvalue_whole'] = sw_whole_pvalue
    _add_diff_ratio_feats(feats, 'shapiro_pvalue', sw1_pvalue, sw2_pvalue)
    _add_contribution_ratio_feats(feats, 'shapiro_pvalue', sw1_pvalue, sw2_pvalue, sw_whole_pvalue)

    # Jarque-Bera检验差异
    jb1_stat, jb1_pvalue, jb2_stat, jb2_pvalue, jb_whole_stat, jb_whole_pvalue = (np.nan,)*6
    try:
        jb1_stat, jb1_pvalue = scipy.stats.jarque_bera(s1)
        jb2_stat, jb2_pvalue = scipy.stats.jarque_bera(s2)
        jb_whole_stat, jb_whole_pvalue = scipy.stats.jarque_bera(s_whole)
    except:
        pass
    feats['jb_pvalue_left'] = jb1_pvalue
    feats['jb_pvalue_right'] = jb2_pvalue
    feats['jb_pvalue_whole'] = jb_whole_pvalue
    _add_diff_ratio_feats(feats, 'jb_pvalue', jb1_pvalue, jb2_pvalue)
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
        _add_diff_ratio_feats(feats, 'kpss_pvalue', k1['p'], k2['p'])
        _add_contribution_ratio_feats(feats, 'kpss_pvalue', k1['p'], k2['p'], k_whole['p'])

        feats['kpss_stat_left'] = k1['stat']
        feats['kpss_stat_right'] = k2['stat']
        feats['kpss_stat_whole'] = k_whole['stat']
        _add_diff_ratio_feats(feats, 'kpss_stat', k1['stat'], k2['stat'])
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
        _add_diff_ratio_feats(feats, 'adf_pvalue', f1['p'], f2['p'])
        _add_contribution_ratio_feats(feats, 'adf_pvalue', f1['p'], f2['p'], f_whole['p'])

        feats['adf_stat_left'] = f1['stat']
        feats['adf_stat_right'] = f2['stat']
        feats['adf_stat_whole'] = f_whole['stat']
        _add_diff_ratio_feats(feats, 'adf_stat', f1['stat'], f2['stat'])
        _add_contribution_ratio_feats(feats, 'adf_stat', f1['stat'], f2['stat'], f_whole['stat'])

        feats['adf_icbest_left'] = f1['ic']
        feats['adf_icbest_right'] = f2['ic']
        feats['adf_icbest_whole'] = f_whole['ic']
        _add_diff_ratio_feats(feats, 'adf_icbest', f1['ic'], f2['ic'])
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

    def analyze_cumsum_curve(series, seg):
        """分析累积和曲线的各种特征"""
        if len(series) < 3:
            return {}
        
        cumsum_curve = series.cumsum()
        curve_feats = {}
        
        # 线性趋势
        x = np.arange(len(cumsum_curve))
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, cumsum_curve)
        curve_feats[f'cumsum_linear_trend_slope_{seg}'] = slope
        curve_feats[f'cumsum_linear_trend_r2_{seg}'] = r_value ** 2
        curve_feats[f'cumsum_linear_trend_pvalue_{seg}'] = p_value

        # 波动率
        curve_feats[f'cumsum_std_{seg}'] = np.std(cumsum_curve)
        curve_feats[f'cumsum_cv_{seg}'] = safe_cv(cumsum_curve)
    
        # 趋势背离
        linear_trend = slope * x + intercept
        detrended = cumsum_curve - linear_trend
        curve_feats[f'cumsum_detrend_volatility_{seg}'] = np.std(detrended)
        curve_feats[f'cumsum_detrend_volatility_normalized_{seg}'] = np.std(detrended) / (np.abs(np.mean(cumsum_curve)) + 1e-6)
        curve_feats[f'cumsum_detrend_max_deviation_{seg}'] = np.max(np.abs(detrended))
        
        # 极值特征
        curve_feats[f'cumsum_min_{seg}'] = np.min(cumsum_curve)
        curve_feats[f'cumsum_max_{seg}'] = np.max(cumsum_curve)
        
        return curve_feats
    
    feats.update(analyze_cumsum_curve(s1, 'left'))
    feats.update(analyze_cumsum_curve(s2, 'right'))
    feats.update(analyze_cumsum_curve(s_whole, 'whole'))
    
    _add_diff_ratio_feats(feats, 'cumsum_linear_trend_slope', feats.get('cumsum_linear_trend_slope_left', 0), feats.get('cumsum_linear_trend_slope_right', 0))
    _add_diff_ratio_feats(feats, 'cumsum_std', feats.get('cumsum_std_left', 0), feats.get('cumsum_std_right', 0))
    _add_diff_ratio_feats(feats, 'cumsum_cv', feats.get('cumsum_cv_left', 0), feats.get('cumsum_cv_right', 0))
    _add_contribution_ratio_feats(feats, 'cumsum_min', feats.get('cumsum_min_left', 0), feats.get('cumsum_min_right', 0), feats.get('cumsum_min_whole', 0))
    _add_contribution_ratio_feats(feats, 'cumsum_max', feats.get('cumsum_max_left', 0), feats.get('cumsum_max_right', 0), feats.get('cumsum_max_whole', 0))

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
    _add_diff_ratio_feats(feats, 'zero_cross', zc1, zc2)
    _add_contribution_ratio_feats(feats, 'zero_cross', zc1, zc2, zc_whole)
    
    def autocorr_lag1(s):
        if len(s) < 2: return 0.0
        ac = s.autocorr(lag=1)
        return ac if not np.isnan(ac) else 0.0
        
    ac1, ac2, ac_whole = autocorr_lag1(s1), autocorr_lag1(s2), autocorr_lag1(s_whole)
    feats['autocorr_lag1_left'] = ac1
    feats['autocorr_lag1_right'] = ac2
    feats['autocorr_lag1_whole'] = ac_whole
    _add_diff_ratio_feats(feats, 'autocorr_lag1', ac1, ac2)
    _add_contribution_ratio_feats(feats, 'autocorr_lag1', ac1, ac2, ac_whole)

    var1, var2, var_whole = s1.diff().var(), s2.diff().var(), s_whole.diff().var()
    feats['diff_var_left'] = var1
    feats['diff_var_right'] = var2
    feats['diff_var_whole'] = var_whole
    _add_diff_ratio_feats(feats, 'diff_var', var1, var2)
    _add_contribution_ratio_feats(feats, 'diff_var', var1, var2, var_whole)
    
    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 5. 频域特征 ---
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
    _add_diff_ratio_feats(feats, 'dominant_freq', freq1, freq2)
    _add_contribution_ratio_feats(feats, 'dominant_freq', freq1, freq2, freq_whole)

    feats['max_power_left'] = power1
    feats['max_power_right'] = power2
    feats['max_power_whole'] = power_whole
    _add_diff_ratio_feats(feats, 'max_power', power1, power2)
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
    _add_diff_ratio_feats(feats, 'ptp', ptp1, ptp2)
    _add_contribution_ratio_feats(feats, 'ptp', ptp1, ptp2, ptp_whole)

    feats['iqr_left'] = iqr1
    feats['iqr_right'] = iqr2
    feats['iqr_whole'] = iqr_whole
    _add_diff_ratio_feats(feats, 'iqr', iqr1, iqr2)
    _add_contribution_ratio_feats(feats, 'iqr', iqr1, iqr2, iqr_whole)
    
    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 7. 熵信息 ---
@register_feature(func_id="7")
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
            _add_diff_ratio_feats(feats, name, v1, v2)
            _add_contribution_ratio_feats(feats, name, v1, v2, v_whole)
        except Exception:
            feats.update({f'{name}_left': 0, f'{name}_right': 0, f'{name}_whole': 0, f'{name}_diff': 0, f'{name}_ratio': 0})

    try:
        m1, c1 = antropy.hjorth_params(s1)
        m2, c2 = antropy.hjorth_params(s2)
        m_whole, c_whole = antropy.hjorth_params(s_whole)
        feats.update({
            'hjorth_mobility_left': m1, 'hjorth_mobility_right': m2, 'hjorth_mobility_whole': m_whole,
            'hjorth_complexity_left': c1, 'hjorth_complexity_right': c2, 'hjorth_complexity_whole': c_whole,
        })
        _add_diff_ratio_feats(feats, 'hjorth_mobility', m1, m2)
        _add_contribution_ratio_feats(feats, 'hjorth_mobility', m1, m2, m_whole)
        _add_diff_ratio_feats(feats, 'hjorth_complexity', c1, c2)
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
        })
        _add_diff_ratio_feats(feats, 'lziv_complexity', lz1, lz2)
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
        })
        _add_diff_ratio_feats(feats, 'cond_entropy', ce1, ce2)
        _add_contribution_ratio_feats(feats, 'cond_entropy', ce1, ce2, ce_whole)
    except Exception:
        feats.update({'cond_entropy_left':0, 'cond_entropy_right':0, 'cond_entropy_whole':0, 'cond_entropy_diff':0, 'cond_entropy_ratio':0})
    
    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 8. tsfresh --- 
@register_feature(func_id="8")
def tsfresh_features(u: pd.DataFrame) -> dict:
    """基于tsfresh的特征工程"""
    s1 = u['value'][u['period'] == 0].to_numpy()
    s2 = u['value'][u['period'] == 1].to_numpy()
    s_whole = u['value'].to_numpy()
    feats = {}

    funcs = {
        tsfresh_fe.ratio_value_number_to_time_series_length: None,
        tsfresh_fe.sum_of_reoccurring_data_points: None,
        tsfresh_fe.percentage_of_reoccurring_values_to_all_values: None,
        tsfresh_fe.percentage_of_reoccurring_datapoints_to_all_datapoints: None,
        tsfresh_fe.last_location_of_maximum: None,
        tsfresh_fe.first_location_of_maximum: None,
        tsfresh_fe.has_duplicate: None,
        tsfresh_fe.benford_correlation: None,
        tsfresh_fe.ratio_beyond_r_sigma: [6, 3, 1.5, 1, 0.5],
        tsfresh_fe.quantile: [0.6, 0.4, 0.1],
        tsfresh_fe.count_above: [0],
        tsfresh_fe.number_peaks: [25, 50],
        tsfresh_fe.partial_autocorrelation: [{"lag": 2}, {"lag": 6}],
        tsfresh_fe.index_mass_quantile: [{"q": 0.1}, {"q": 0.6}, {"q": 0.7}, {"q": 0.8}],
        tsfresh_fe.ar_coefficient: [{"coeff": 0, "k": 10}, {"coeff": 2, "k": 10}, {"coeff": 8, "k": 10}],
        tsfresh_fe.linear_trend: [{"attr": "slope"}, {"attr": "rvalue"}, {"attr": "pvalue"}, {"attr": "intercept"}],
        tsfresh_fe.fft_coefficient: [{"coeff": 3, "attr": "imag"}, {"coeff": 2, "attr": "imag"}, {"coeff": 1, "attr": "imag"}],
        tsfresh_fe.energy_ratio_by_chunks: [{"num_segments": 10, "segment_focus": 9}],
        tsfresh_fe.friedrich_coefficients: [{"m": 3, "r": 30, "coeff": 2}, {"m": 3, "r": 30, "coeff": 3}],
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
        ],
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
                    _add_diff_ratio_feats(feats, feat_name_base, v1, v2)
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
                _add_diff_ratio_feats(feats, base_name, v1, v2)
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

# --- 9. 时间序列建模 ---
@register_feature(func_id="9")
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
    _add_diff_ratio_feats(feats, 'ar_resid_std', s1_resid_std, s2_resid_std)
    _add_contribution_ratio_feats(feats, 'ar_resid_std', s1_resid_std, s2_resid_std, swhole_resid_std)
    
    feats['ar_aic_left'] = s1_aic
    feats['ar_aic_right'] = s2_aic
    feats['ar_aic_whole'] = swhole_aic
    _add_diff_ratio_feats(feats, 'ar_aic', s1_aic, s2_aic)
    _add_contribution_ratio_feats(feats, 'ar_aic', s1_aic, s2_aic, swhole_aic)

    feats['ar_bic_left'] = s1_bic
    feats['ar_bic_right'] = s2_bic
    feats['ar_bic_whole'] = swhole_bic
    _add_diff_ratio_feats(feats, 'ar_bic', s1_bic, s2_bic)
    _add_contribution_ratio_feats(feats, 'ar_bic', s1_bic, s2_bic, swhole_bic)
    
    # 比较模型系数
    for i in range(len(s1_params)):
        feats[f'ar_param_{i}_left'] = s1_params[i]
        feats[f'ar_param_{i}_right'] = s2_params[i]
        feats[f'ar_param_{i}_whole'] = swhole_params[i]
        _add_diff_ratio_feats(feats, f'ar_param_{i}', s1_params[i], s2_params[i])
        _add_contribution_ratio_feats(feats, f'ar_param_{i}', s1_params[i], s2_params[i], swhole_params[i])

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 10. 分段损失 ---
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
                # diff = right - left if left is not None and right is not None else None
                # ratio = right / (left + 1e-6) if left is not None and right is not None else None
            except Exception:
                left = None
                right = None
                whole = None
                # diff = None
                # ratio = None
            # Move to _add_diff_ratio_feats, 'diff': diff, 'ratio': ratio
            result[name] = {'left': left, 'right': right, 'whole': whole}
        return result

@register_feature(func_id="10")
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
        _add_diff_ratio_feats(feats, f'rpt_cost_{k}', v['left'], v['right'])
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

# @register_transform(output_mode_names=['MAtrend', 'MAresid'])
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

def safe_cvstd_vectorized(values, window_size=50):
    """
    向量化计算滑动变异系数*标准差，比逐个窗口计算更快
    
    Args:
        values: numpy array，输入序列
        window_size: int，窗口大小
        
    Returns:
        numpy array: 变异系数*标准差值序列
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    
    # 使用pandas的rolling功能进行向量化计算
    series = pd.Series(values)
    
    # 计算滑动均值和标准差
    rolling_mean = series.rolling(window=window_size, center=True, min_periods=1).mean()
    rolling_std = series.rolling(window=window_size, center=True, min_periods=1).std()
    
    # 计算变异系数*标准差，避免除零
    result = np.zeros(n)
    mask = np.abs(rolling_mean) > 1e-6
    result[mask] = (rolling_std[mask] * rolling_std[mask]) / rolling_mean[mask]
    result[~mask] = 0.0
    
    # 处理边界值
    half_window = window_size // 2
    if half_window > 0:
        result[:half_window] = result[half_window]
        result[-half_window:] = result[-half_window]
    
    return result

# @register_transform(output_mode_names=['MCS'])
def moving_cv_std(X_df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    滑动变异系数*标准差
    
    Args:
        X_df: 输入数据框，包含MultiIndex (id, time) 和 columns ['value', 'period']
        
    Returns:
        List[pd.DataFrame]: 包含数据框的列表 [变异系数*标准差值]
    """
    X_df_sorted = X_df.sort_index()
    result_dfs = []
    
    # 为每个模态创建一个空的数据框
    for mode_name in ['cvstd']:
        mode_df = X_df_sorted.copy()
        mode_df['value'] = np.nan
        result_dfs.append(mode_df)
    
    # 获取所有唯一的series_id
    unique_ids = X_df_sorted.index.get_level_values('id').unique()
    
    # 对每个id进行分解，添加进度条
    for series_id in tqdm(unique_ids, desc="计算滑动变异系数*标准差", unit="series"):
        series_data = X_df_sorted.loc[series_id]
        series_data = series_data.sort_index()
        values = series_data['value'].values
        
        # 使用向量化版本进行快速计算
        window_size = 50
        cvstd_values = safe_cvstd_vectorized(values, window_size)
        
        result_dfs[0].loc[series_id, 'value'] = cvstd_values  # 变异系数*标准差值
    
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

def _load_feature_dict_file(file_path: Path):
    """加载字典格式的特征文件及其元数据。"""
    if not file_path or not file_path.exists():
        return {}, {}
    try:
        # 加载主文件获取元数据
        main_table = pd.read_parquet(file_path)
        metadata_str = main_table.attrs.get('feature_metadata', '{}')
        metadata = json.loads(metadata_str)
        
        # 加载字典格式的特征数据
        feature_dict = {}
        base_name = file_path.stem  # 去掉扩展名
        
        # 查找所有相关的特征文件
        for data_id_file in file_path.parent.glob(f"{base_name}_id_*.parquet"):
            # 从文件名提取数据ID
            data_id = data_id_file.stem.split('_id_')[-1]
            feature_dict[data_id] = pd.read_parquet(data_id_file)
        
        # 如果没有找到分离的文件，尝试从主文件加载（向后兼容）
        if not feature_dict and not main_table.empty:
            feature_dict["0"] = main_table
            
        return feature_dict, metadata
    except Exception as e:
        logger.warning(f"无法加载字典格式特征文件 {file_path}: {e}。")
        return {}, {}

def _save_feature_file(df: pd.DataFrame, metadata: dict) -> Path:
    """将特征保存到一个新的、带时间戳的文件中，并返回其路径。"""
    config.FEATURE_DIR.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_feature_path = config.FEATURE_DIR / f'features_{timestamp}.parquet'
    
    df.attrs['feature_metadata'] = json.dumps(metadata)
    df.to_parquet(new_feature_path)
    return new_feature_path

def _save_feature_dict_file(feature_dict: dict, metadata: dict) -> Path:
    """将字典格式的特征保存到带时间戳的文件中，并返回主文件路径。"""
    config.FEATURE_DIR.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f'features_{timestamp}'
    main_feature_path = config.FEATURE_DIR / f'{base_name}.parquet'
    
    # 保存每个数据ID的特征文件
    for data_id, df in feature_dict.items():
        if not df.empty:
            data_id_path = config.FEATURE_DIR / f'{base_name}_id_{data_id}.parquet'
            df.to_parquet(data_id_path)
    
    # 创建主文件用于存储元数据（可以是空的DataFrame）
    main_df = pd.DataFrame()
    main_df.attrs['feature_metadata'] = json.dumps(metadata)
    main_df.to_parquet(main_feature_path)
    
    return main_feature_path

def _backup_feature_file(file_path: Path):
    """
    备份特征文件及其所有相关的增强数据文件。
    
    Args:
        file_path: 基础特征文件路径（例如 features_20250724_232149.parquet）
    """
    if not file_path or not file_path.exists():
        return
    
    config.FEATURE_BACKUP_DIR.mkdir(exist_ok=True, parents=True)
    
    # 备份基础文件
    backup_path = config.FEATURE_BACKUP_DIR / file_path.name
    file_path.rename(backup_path)
    logger.info(f"已将基础文件 {file_path.name} 备份到: {config.FEATURE_BACKUP_DIR}")
    
    # 查找并备份所有相关的增强数据文件
    # 从基础文件名中提取时间戳部分
    base_name = file_path.stem  # 去掉 .parquet 后缀
    if base_name.startswith('features_'):
        # 查找所有匹配的增强数据文件
        pattern = f"{base_name}_id_*.parquet"
        feature_dir = file_path.parent
        
        # 使用 glob 查找所有匹配的文件
        related_files = list(feature_dir.glob(pattern))
        
        if related_files:
            logger.info(f"找到 {len(related_files)} 个相关的增强数据文件")
            for related_file in related_files:
                related_backup_path = config.FEATURE_BACKUP_DIR / related_file.name
                related_file.rename(related_backup_path)
                logger.info(f"已将增强数据文件 {related_file.name} 备份到: {config.FEATURE_BACKUP_DIR}")
        else:
            logger.info("未找到相关的增强数据文件")
    else:
        logger.warning(f"文件名格式不符合预期: {file_path.name}")

def _apply_feature_func_sequential(func, X_df: pd.DataFrame, use_tqdm: bool = False) -> pd.DataFrame:
    """顺序应用单个特征函数"""
    all_ids = X_df.index.get_level_values("id").unique()
    iterator = (
        tqdm(all_ids, desc=f"Running {func.__name__} (sequentially)")
        if use_tqdm else all_ids
    )
    results = [
        {**{'id': id_val}, **func(X_df.loc[id_val])}
        for id_val in iterator
    ]
    return pd.DataFrame(results).set_index('id')

def _apply_feature_func_parallel(func, X_df: pd.DataFrame) -> pd.DataFrame:
    """并行应用单个特征函数"""
    all_ids = X_df.index.get_level_values("id").unique()
    results = Parallel(n_jobs=config.N_JOBS)(
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
        X_data, 
        funcs_to_run: list = None, 
        trans_to_run: list = None, 
        base_feature_file: str = None
    ):
    """
    生成指定的特征，或者如果未指定，则生成所有已注册的特征。
    可以基于一个现有的特征文件进行增量更新。
    现在支持字典格式的输入数据和特征存储。

    Args:
        X_data: 输入数据，可以是:
            - pd.DataFrame: 单个数据框（向后兼容）
            - dict: 字典格式，键为数据ID（"0"表示原始数据，"1"、"2"等表示增强数据），值为对应的数据框
        funcs_to_run (list, optional): 要运行的特征函数名称列表。
            如果为 None，则运行所有在 `FEATURE_REGISTRY` 中注册的、且不在 `EXPERIMENTAL_FEATURES` 中的函数。
        trans_to_run (list, optional): 要运行的变换函数名称列表。
        base_feature_file (str, optional): 基础特征文件名。如果提供，
            将加载此文件并在此基础上添加或更新特征。否则，将创建一个新的特征集。
    """
    utils.ensure_feature_dirs()
    
    # 处理输入数据格式
    if isinstance(X_data, pd.DataFrame):
        # 向后兼容：单个数据框转换为字典格式
        X_data_dict = {"0": X_data}
        logger.info("输入为单个数据框，已转换为字典格式（数据ID: '0'）")
    elif isinstance(X_data, dict):
        X_data_dict = X_data
        logger.info(f"输入为字典格式，包含数据ID: {list(X_data_dict.keys())}")
    else:
        raise ValueError("X_data必须是pd.DataFrame或dict类型")
    
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

    # 2. 加载基础特征（字典格式）
    if base_path and base_path.exists():
        logger.info(f"将基于特征文件进行更新: {base_path.name}")
        feature_dict, metadata = _load_feature_dict_file(base_path)
        # 如果加载失败，尝试旧格式
        if not feature_dict:
            logger.info("尝试加载旧格式特征文件...")
            old_feature_df, metadata = _load_feature_file(base_path)
            if not old_feature_df.empty:
                feature_dict = {"0": old_feature_df}
    else:
        logger.info("未找到基础特征文件，将创建全新的特征集。")
        feature_dict, metadata = {}, {}
    
    # 确保每个数据ID都有对应的特征DataFrame
    for data_id in X_data_dict.keys():
        if data_id not in feature_dict:
            # 获取该数据ID的唯一ID列表
            unique_ids = X_data_dict[data_id].index.get_level_values('id').unique()
            feature_dict[data_id] = pd.DataFrame(index=unique_ids)
            logger.info(f"为数据ID '{data_id}' 创建新的特征DataFrame，包含 {len(unique_ids)} 个样本")
    
    logger.info(f"基础特征字典包含数据ID: {list(feature_dict.keys())}")
    for data_id, df in feature_dict.items():
        logger.info(f"  数据ID '{data_id}': {df.shape}")

    # 3. 为每个数据ID生成特征
    initial_feature_counts = {data_id: len(df.columns) for data_id, df in feature_dict.items()}
    
    for data_id, X_df in X_data_dict.items():
        logger.info(f"=== 开始为数据ID '{data_id}' 生成特征 ===")
        
        # 时序分解
        logger.info(f"--- 开始时序分解（数据ID: {data_id}） ---")
        transformed_data = apply_transformation(X_df, trans_to_run)
        logger.info(f"分解完成，共生成 {len(transformed_data)} 个模态: {list(transformed_data.keys())}")
        
        # 获取当前数据ID的特征DataFrame
        current_feature_df = feature_dict[data_id]
        loaded_features = current_feature_df.columns.tolist()
        
        # 逐个生成新特征并更新
        for mode_name, mode_df in transformed_data.items():
            logger.info(f"=== 开始为数据ID '{data_id}' 的模态 '{mode_name}' 生成特征 ===")
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
                new_features_df = clean_feature_names(new_features_df)

                # 记录日志
                duration = time.time() - start_time
                logger.info(f"'{func_name}' 生成完毕，耗时: {duration:.2f} 秒。")
                logger.info(f"  新生成特征列名: {new_features_df.columns.tolist()}")
                
                for col in new_features_df.columns:
                    null_ratio = new_features_df[col].isnull().sum() / len(new_features_df)
                    zero_ratio = (new_features_df[col] == 0).sum() / len(new_features_df)
                    logger.info(f"    - '{col}': 空值比例={null_ratio:.2%}, 零值比例={zero_ratio:.2%}")

                # 删除旧版本特征（如果存在），然后合并
                current_feature_df = current_feature_df.drop(columns=new_features_df.columns, errors='ignore')
                current_feature_df = current_feature_df.merge(new_features_df, left_index=True, right_index=True, how='left')
                loaded_features = current_feature_df.columns.tolist()
        
        # 更新特征字典
        feature_dict[data_id] = current_feature_df
        logger.info(f"数据ID '{data_id}' 特征生成完成，最终特征数: {len(current_feature_df.columns)}")

    # 4. 保存结果
    # 检查是否有新特征生成
    has_new_features = False
    final_feature_counts = {}
    
    for data_id, df in feature_dict.items():
        final_feature_counts[data_id] = len(df.columns)
        if final_feature_counts[data_id] > initial_feature_counts[data_id]:
            has_new_features = True
    
    if has_new_features:
        # 只有在成功生成新特征后，才备份旧文件
        if base_path and base_path.exists():
            _backup_feature_file(base_path)
            
        metadata['last_updated_funcs'] = funcs_to_run
        metadata['data_ids'] = list(feature_dict.keys())
        new_file_path = _save_feature_dict_file(feature_dict, metadata)
        logger.info(f"特征已保存到新文件: {new_file_path.name}")
        
        logger.info("--- 生成后完整特征统计 ---")
        for data_id, df in feature_dict.items():
            logger.info(f"数据ID '{data_id}': {len(df.columns)} 个特征")
            logger.info(f"  特征列表: {df.columns.tolist()[:10]}{'...' if len(df.columns) > 10 else ''}")
        logger.info("-----------------------------")
    else:
        logger.info("没有新特征生成，文件未保存。")
        new_file_path = None

    logger.info("=== 特征生成完成统计 ===")
    for data_id in feature_dict.keys():
        initial_count = initial_feature_counts[data_id]
        final_count = final_feature_counts[data_id]
        logger.info(f"数据ID '{data_id}': {initial_count} -> {final_count} 个特征 (+{final_count - initial_count})")
    
    logger.info(f"生成/更新完成。新文件: {new_file_path.name if new_file_path else 'N/A'}")

def delete_features(base_feature_file: str, funcs_to_delete: list = None, cols_to_delete: list = None):
    """
    从指定的特征文件中删除特征，并生成一个新的带时间戳的文件。
    可以按函数名删除，也可以按特定的列名删除。
    支持字典格式的特征数据。
    """
    base_path = config.FEATURE_DIR / base_feature_file
    if not base_path.exists():
        logger.error(f"指定的特征文件不存在: {base_feature_file}")
        return

    logger.info(f"将从文件 {base_feature_file} 中删除特征...")
    
    # 尝试加载字典格式的特征文件
    try:
        feature_dict, metadata = _load_feature_dict_file(base_path)
        is_dict_format = True
        logger.info(f"加载字典格式特征文件，包含数据ID: {list(feature_dict.keys())}")
    except Exception:
        # 回退到旧格式
        feature_df, metadata = _load_feature_file(base_path)
        if feature_df.empty:
            logger.error(f"无法从 {base_feature_file} 加载数据，操作中止。")
            return
        feature_dict = {"0": feature_df}
        is_dict_format = False
        logger.info("加载旧格式特征文件，转换为字典格式处理")

    _backup_feature_file(base_path)
    
    # 为每个数据ID处理特征删除
    updated_feature_dict = {}
    total_deleted_features = {}
    
    for data_id, feature_df in feature_dict.items():
        logger.info(f"\n处理数据ID '{data_id}' 的特征删除...")
        initial_columns = set(feature_df.columns)
        cols_to_drop = []

        if funcs_to_delete:
            logger.info(f"  按函数名删除: {funcs_to_delete}")
            for func_name in funcs_to_delete:
                # 基于函数名约定来匹配列名
                matched_cols = [col for col in feature_df.columns if col.startswith(func_name)]
                if matched_cols:
                    logger.info(f"    函数 '{func_name}' 匹配到 {len(matched_cols)} 列: {matched_cols}")
                    cols_to_drop.extend(matched_cols)
                else:
                    logger.warning(f"    未找到与函数 '{func_name}' 相关的特征列。")

        if cols_to_delete:
            logger.info(f"  按列名删除: {cols_to_delete}")
            # 验证列名是否存在
            valid_cols = [c for c in cols_to_delete if c in feature_df.columns]
            invalid_cols = set(cols_to_delete) - set(valid_cols)
            if invalid_cols:
                logger.warning(f"    以下列名不存在，将被忽略: {list(invalid_cols)}")
            cols_to_drop.extend(valid_cols)

        # 去重并执行删除
        final_cols_to_drop = sorted(list(set(cols_to_drop)))
        if final_cols_to_drop:
            logger.info(f"  数据ID '{data_id}' 将删除 {len(final_cols_to_drop)} 个特征列: {final_cols_to_drop}")
            feature_df_copy = feature_df.copy()
            feature_df_copy.drop(columns=final_cols_to_drop, inplace=True)
            updated_feature_dict[data_id] = feature_df_copy
            total_deleted_features[data_id] = final_cols_to_drop
        else:
            logger.info(f"  数据ID '{data_id}' 没有找到要删除的特征列")
            updated_feature_dict[data_id] = feature_df.copy()
            total_deleted_features[data_id] = []
    
    # 检查是否有任何特征被删除
    total_deleted_count = sum(len(deleted) for deleted in total_deleted_features.values())
    if total_deleted_count == 0:
        logger.warning("没有找到任何要删除的特征列，操作中止。")
        # 恢复备份，因为没有变化
        backup_path = config.FEATURE_BACKUP_DIR / base_path.name
        if backup_path.exists():
            backup_path.rename(base_path)
            logger.info("已恢复原始文件。")
        return
    
    # 更新元数据
    metadata['last_deleted_features'] = total_deleted_features
    metadata['data_ids'] = list(updated_feature_dict.keys())
    
    # 保存结果
    if is_dict_format or len(updated_feature_dict) > 1:
        new_path = _save_feature_dict_file(updated_feature_dict, metadata)
    else:
        # 如果原来是单个DataFrame格式且只有一个数据ID，保持兼容性
        new_path = _save_feature_file(updated_feature_dict["0"], metadata)
    
    logger.info(f"\n=== 特征删除完成统计 ===")
    for data_id, deleted_features in total_deleted_features.items():
        remaining_count = len(updated_feature_dict[data_id].columns)
        logger.info(f"数据ID '{data_id}': 删除了 {len(deleted_features)} 个特征，剩余 {remaining_count} 个特征")
    
    logger.info(f"删除完成。新文件: {new_path.name}, 总计删除 {total_deleted_count} 个特征")


def load_features(feature_file: str = None, data_ids: list = None) -> tuple[pd.DataFrame | None, str | None]:
    """加载指定的或最新的特征文件，并拼接指定数据ID的特征数据。
    
    Args:
        feature_file (str, optional): 特征文件名。如果未指定，将加载最新版本。
        data_ids (list, optional): 要使用的数据ID列表，例如["0", "1"]。如果未指定，默认使用["0"]。
    
    Returns:
        tuple: (拼接后的特征数据, 文件名) 或 (None, None)
    """
    # 使用一个临时的logger，避免依赖全局logger
    import logging
    temp_logger = logging.getLogger('load_features')
    if not temp_logger.handlers:
        temp_logger.addHandler(logging.StreamHandler())
        temp_logger.setLevel(logging.INFO)

    if feature_file:
        path_to_load = config.FEATURE_DIR / feature_file
    else:
        temp_logger.info("未指定特征文件，将尝试加载最新版本。")
        path_to_load = _get_latest_feature_file()

    if not path_to_load or not path_to_load.exists():
        temp_logger.error(f"无法找到要加载的特征文件: {path_to_load}")
        return None, None

    temp_logger.info(f"正在从 {path_to_load.name} 加载特征...")
    
    # 如果未指定data_ids，默认使用["0"]
    if data_ids is None:
        data_ids = ["0"]
    
    # 尝试加载字典格式的特征文件
    try:
        feature_dict, _ = _load_feature_dict_file(path_to_load)
        temp_logger.info(f"加载字典格式特征文件成功，包含数据ID: {list(feature_dict.keys())}")
        
        # 检查请求的数据ID是否存在
        available_ids = list(feature_dict.keys())
        missing_ids = [id for id in data_ids if id not in available_ids]
        if missing_ids:
            temp_logger.warning(f"请求的数据ID {missing_ids} 在特征文件中不存在，可用的ID: {available_ids}")
            # 只使用存在的ID
            data_ids = [id for id in data_ids if id in available_ids]
            if not data_ids:
                temp_logger.error("没有可用的数据ID")
                return None, None
        
        # 拼接指定数据ID的特征数据
        feature_dfs = []
        for data_id in data_ids:
            df = feature_dict[data_id].copy()
            feature_dfs.append(df)
        
        # 按行拼接（concat along axis=0），保持特征列数不变
        if len(feature_dfs) == 1:
            concatenated_df = feature_dfs[0]
        else:
            concatenated_df = pd.concat(feature_dfs, axis=0, ignore_index=False)
        
        total_features = len(concatenated_df.columns)
        total_rows = len(concatenated_df)
        temp_logger.info(f"特征拼接成功，使用数据ID: {data_ids}，共 {total_features} 个特征，{total_rows} 行数据。")
        return concatenated_df, path_to_load.name
                
    except Exception:
        # 回退到旧格式
        feature_df, _ = _load_feature_file(path_to_load)
        
        if feature_df.empty:
            return None, None
        
        # 对于旧格式，只能返回单个DataFrame（相当于数据ID "0"）
        if "0" in data_ids:
            temp_logger.info(f"特征加载成功（旧格式），共 {len(feature_df.columns)} 个特征。")
            return feature_df, path_to_load.name
        else:
            temp_logger.warning(f"旧格式特征文件只支持数据ID '0'，但请求的是 {data_ids}")
            return None, None
