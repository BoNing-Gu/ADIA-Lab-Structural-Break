import pandas as pd
import numpy as np
import scipy.stats
import statsmodels as sm
import statsmodels.tsa.api as tsa
import antropy
from tsfresh.feature_extraction import feature_calculators as tsfresh_fe

import re
import json
import time
import logging
import inspect
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
from joblib import Parallel, delayed
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

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


warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", InterpolationWarning)
warnings.simplefilter("ignore", FutureWarning)

# logger 将由 main.py 在运行时注入
logger = None

# --- 特征函数注册表 ---
FEATURE_REGISTRY = {}

def register_feature(_func=None, *, parallelizable=True):
    """一个用于注册特征函数的装饰器，可以标记特征是否可并行化。"""
    def decorator_register(func):
        FEATURE_REGISTRY[func.__name__] = {"func": func, "parallelizable": parallelizable}
        return func

    if _func is None:
        # Used as @register_feature(parallelizable=...)
        return decorator_register
    else:
        # Used as @register_feature
        return decorator_register(_func)

# --- 1. 分布统计特征 ---
@register_feature
def distribution_stats_features(u: pd.DataFrame) -> dict:
    """统计量的Diff"""
    s1 = u['value'][u['period'] == 0]
    s2 = u['value'][u['period'] == 1]
    feats = {}

    mean1, mean2 = s1.mean(), s2.mean()
    feats['mean_diff'] = mean2 - mean1

    median_diff = np.median(s1) - np.median(s2)
    feats['median_diff'] = median_diff if not np.isnan(median_diff) else 0

    max_diff = np.max(s1) - np.max(s2)
    feats['max_diff'] = max_diff if not np.isnan(max_diff) else 0
    
    min_diff = np.min(s1) - np.min(s2)
    feats['min_diff'] = min_diff if not np.isnan(min_diff) else 0
    
    range1 = np.max(s1) - np.min(s1)
    range2 = np.max(s2) - np.min(s2)
    feats['range_diff'] = range1 - range2 if not (np.isnan(range1) or np.isnan(range2)) else 0
    
    std1, std2 = s1.std(), s2.std()
    feats['std_diff'] = std2 - std1
    if std1 > 1e-6:
        feats['std_ratio'] = std2 / std1
    else:
        feats['std_ratio'] = 1.0 if std2 < 1e-6 else 1e6
    
    feats['skew_diff'] = s2.skew() - s1.skew()
    feats['kurt_diff'] = s2.kurt() - s1.kurt()

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
    
# --- 2. 假设检验统计量特征 ---
@register_feature
def test_stats_features(u: pd.DataFrame) -> dict:
    """假设检验统计量 & 分段假设检验统计量的Diff"""
    s1 = u['value'][u['period'] == 0]
    s2 = u['value'][u['period'] == 1]
    feats = {}

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
    
    # Shapiro-Wilk检验
    if len(s1) <= 5000 and len(s2) <= 5000:  # Shapiro有样本大小限制
        sw1_stat, sw1_pvalue = scipy.stats.shapiro(s1)
        sw2_stat, sw2_pvalue = scipy.stats.shapiro(s2)
        feats['shapiro_pvalue_0'] = sw1_pvalue
        feats['shapiro_pvalue_1'] = sw2_pvalue
        feats['shapiro_pvalue_diff'] = sw1_pvalue - sw2_pvalue if not (np.isnan(sw1_pvalue) or np.isnan(sw2_pvalue)) else 0
    else:
        feats['shapiro_pvalue_0'] = 1
        feats['shapiro_pvalue_1'] = 1
        feats['shapiro_pvalue_diff'] = 0

    # Jarque-Bera检验差异
    try:
        jb1_stat, jb1_pvalue = scipy.stats.jarque_bera(s1)
        jb2_stat, jb2_pvalue = scipy.stats.jarque_bera(s2)
        feats['jb_pvalue_0'] = jb1_pvalue
        feats['jb_pvalue_1'] = jb2_pvalue
        feats['jb_pvalue_diff'] = jb1_pvalue - jb2_pvalue if not (np.isnan(jb1_pvalue) or np.isnan(jb2_pvalue)) else 0
    except:
        feats['jb_pvalue_0'] = 1
        feats['jb_pvalue_1'] = 1
        feats['jb_pvalue_diff'] = 0

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
            'reject_5pct': int(stat > crit_5pct)  # KPSS原假设是“平稳”，所以 > 临界值 拒绝平稳
        }
    try:
        k1 = extract_kpss_features(s1)
        k2 = extract_kpss_features(s2)

        feats['kpss_pvalue_0'] = k1['p']
        feats['kpss_pvalue_1'] = k2['p']
        feats['kpss_pvalue_diff'] = k1['p'] - k2['p']
        feats['kpss_stat_diff'] = k1['stat'] - k2['stat']
    except:
        feats['kpss_pvalue_0'] = 1
        feats['kpss_pvalue_1'] = 1
        feats['kpss_pvalue_diff'] = 0
        feats['kpss_stat_diff'] = 0

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

        feats['adf_pvalue_0'] = f1['p']
        feats['adf_pvalue_1'] = f2['p']
        feats['adf_pvalue_diff'] = f1['p'] - f2['p']
        feats['adf_stat_diff'] = f1['stat'] - f2['stat']
        feats['adf_icbest_diff'] = f1['ic'] - f2['ic']
    except:
        feats['adf_pvalue_0'] = 1
        feats['adf_pvalue_1'] = 1
        feats['adf_pvalue_diff'] = 0
        feats['adf_stat_diff'] = 0
        feats['adf_icbest_diff'] = 0

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 3. 累积和特征 ---
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

# --- 4. 振荡特征 ---
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

# --- 5. 周期性特征 ---
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

# --- 6. 振幅特征 ---
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

# --- 7. 波动性的波动性特征 ---
@register_feature
def volatility_of_volatility_features(u: pd.DataFrame) -> dict:
    """
    计算滚动标准差序列的统计特征，以捕捉“波动性的波动性”的变化。
    在Period 0和Period 1内部，分别计算小窗口（如长度为50）的滚动标准差，
    然后比较这两条新的滚动标准差序列的均值，生成四个相关特征：
    1. Period 0 的滚动标准差均值
    2. Period 1 的滚动标准差均值
    3. 两者之差
    4. 两者之比
    """
    s1 = u['value'][u['period'] == 0]
    s2 = u['value'][u['period'] == 1]
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

    feats[f'rolling_std_w{window}_mean_diff'] = mean2 - mean1
    

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 8. 时间序列建模 ---
@register_feature
def ar_model_features(u: pd.DataFrame) -> dict:
    s1 = u['value'][u['period'] == 0].reset_index(drop=True)
    s2 = u['value'][u['period'] == 1].reset_index(drop=True)
    feats = {}

    def fit_ar(s, lags=10):
        if len(s) <= lags + 1:
            return None
        try:
            # from statsmodels.tsa.ar_model import AutoReg
            return tsa.AutoReg(s, lags=lags, old_names=False).fit()
        except Exception:
            return None

    lags = 10
    model1 = fit_ar(s1, lags)
    model2 = fit_ar(s2, lags)

    # 残差标准差 & AIC 差异
    if model1 is not None and model2 is not None:
        feats['ar_resid_std_diff'] = model2.resid.std() - model1.resid.std()
        feats['ar_aic_diff'] = model2.aic - model1.aic
        feats['ar_const_diff'] = model2.params['const'] - model1.params['const']
        for i in range(1, lags + 1):
            feats[f'ar_param_diff_{i}'] = model2.params[f'value.L{i}'] - model1.params[f'value.L{i}']
    else:
        feats['ar_resid_std_diff'] = 0.0
        feats['ar_aic_diff'] = 0.0
        feats['ar_const_diff'] = 0.0
        for i in range(1, lags + 1):
            feats[f'ar_param_diff_{i}'] = 0.0

    # period=0 拟合后预测 period=1 前 len(s2) 步
    if model1 is not None and len(s2) > 0:
        try:
            max_lag = max(model1.model.ar_lags)
            # 获取 period=0 的尾部作为预测初值
            history = s1.iloc[-max_lag:].tolist()
            preds = []

            for t in range(len(s2)):
                lagged_vals = history[-max_lag:]
                pred = model1.params['const'] if 'const' in model1.params else 0.0
                for i, lag in enumerate(model1.model.ar_lags):
                    pred += model1.params[f'value.L{lag}'] * lagged_vals[-lag]
                preds.append(pred)
                history.append(s2.iloc[t])  # 模拟滚动更新

            preds = np.array(preds)
            mse = np.mean((preds - s2.values[:len(preds)]) ** 2)
            feats['ar_predict_mse'] = mse
        except Exception as e:
            print(f"[WARN] Prediction error: {e}")
            feats['ar_predict_mse'] = 0.0
    else:
        feats.update({'arima_residuals_s2_pred_mean': 0, 'arima_residuals_s2_pred_std': 0, 'arima_residuals_s2_pred_skew': 0, 'arima_residuals_s2_pred_kurt': 0})

    # --- 特征组2: 用 s2 训练，预测 s1 ---
    if len(s2_pd) > min_len and len(s1_pd) > 0:
        try:
            sf2 = StatsForecast(models=[ARIMA(order=order, season_length=0)], freq=1)
            sf2.fit(df2)
            forecasts_on_s1 = sf2.predict(h=len(s1_pd))
            predictions_on_s1 = forecasts_on_s1['ARIMA'].to_cupy().get()
            residuals_s1_pred = s1_pd.values - predictions_on_s1
            feats['arima_residuals_s1_pred_mean'] = np.mean(residuals_s1_pred)
            feats['arima_residuals_s1_pred_std'] = np.std(residuals_s1_pred)
            feats['arima_residuals_s1_pred_skew'] = pd.Series(residuals_s1_pred).skew()
            feats['arima_residuals_s1_pred_kurt'] = pd.Series(residuals_s1_pred).kurt()
        except Exception:
            feats.update({'arima_residuals_s1_pred_mean': 0, 'arima_residuals_s1_pred_std': 0, 'arima_residuals_s1_pred_skew': 0, 'arima_residuals_s1_pred_kurt': 0})
    else:
        feats.update({'arima_residuals_s1_pred_mean': 0, 'arima_residuals_s1_pred_std': 0, 'arima_residuals_s1_pred_skew': 0, 'arima_residuals_s1_pred_kurt': 0})

    # --- 特征组3: 分别建模，比较差异 ---
    s1_resid_std, s2_resid_std = np.nan, np.nan
    s1_params, s2_params = {}, {}

    if len(s1_pd) > min_len:
        try:
            sf1 = StatsForecast(models=[ARIMA(order=order, season_length=0)], freq=1)
            fitted_vals_df1 = sf1.fit(df1).predict(h=0, fitted=True)
            if not fitted_vals_df1.empty:
                fitted_vals1 = fitted_vals_df1['ARIMA'].to_cupy().get()
                s1_resid = s1_pd.values[-len(fitted_vals1):] - fitted_vals1
                s1_resid_std = np.std(s1_resid)
                params_cupy = sf1.models[0].model_
                s1_params = {k: v.get() if hasattr(v, 'get') else v for k,v in params_cupy.items()}
        except Exception:
            pass

    if len(s2_pd) > min_len:
        try:
            sf2 = StatsForecast(models=[ARIMA(order=order, season_length=0)], freq=1)
            fitted_vals_df2 = sf2.fit(df2).predict(h=0, fitted=True)
            if not fitted_vals_df2.empty:
                fitted_vals2 = fitted_vals_df2['ARIMA'].to_cupy().get()
                s2_resid = s2_pd.values[-len(fitted_vals2):] - fitted_vals2
                s2_resid_std = np.std(s2_resid)
                params_cupy = sf2.models[0].model_
                s2_params = {k: v.get() if hasattr(v, 'get') else v for k,v in params_cupy.items()}
        except Exception:
            pass
            
    feats['arima_resid_std_diff'] = (s2_resid_std - s1_resid_std) if not (np.isnan(s1_resid_std) or np.isnan(s2_resid_std)) else 0
    
    all_param_names = sorted(list(set(s1_params.keys()) | set(s2_params.keys())))
    if 'constant' in all_param_names: all_param_names.remove('constant')
    
    # 统一AR和MA系数的名称和长度
    max_ar = max(len(s1_params.get('ar', [])), len(s2_params.get('ar', [])))
    max_ma = max(len(s1_params.get('ma', [])), len(s2_params.get('ma', [])))

    p1_ar = np.resize(s1_params.get('ar', []), max_ar)
    p2_ar = np.resize(s2_params.get('ar', []), max_ar)
    p1_ma = np.resize(s1_params.get('ma', []), max_ma)
    p2_ma = np.resize(s2_params.get('ma', []), max_ma)

    for i in range(max_ar):
        feats[f'arima_param_ar_{i+1}_diff'] = p2_ar[i] - p1_ar[i]
    for i in range(max_ma):
        feats[f'arima_param_ma_{i+1}_diff'] = p2_ma[i] - p1_ma[i]

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}


@register_feature
def arima_model_features(u: pd.DataFrame) -> dict:
    """
    ARIMA 特征提取的动态调度器。
    如果检测到GPU，则使用`StatsForecast`进行加速计算，否则回退到
    使用`statsmodels`的CPU版本。
    """
    if GPU_AVAILABLE:
        # logger is not available here, print for now
        # logger.info("GPU detected, using arima_model_features_gpu.")
        return arima_model_features_gpu(u)
    else:
        # logger.info("No GPU detected, using arima_model_features_cpu.")
        return arima_model_features_cpu(u)


# --- 10. 熵信息 ---
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

# --- 11. 分形 ---
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

# --- 11. tsfresh --- 
@register_feature
def tsfresh_features(u: pd.DataFrame) -> dict:
    """基于tsfresh的特征工程"""
    s1 = u['value'][u['period'] == 0].to_numpy()
    s2 = u['value'][u['period'] == 1].to_numpy()
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

    def cal_func_diff_as_feature(func, param=None):
        if param is None:
            # Simple function: return a scalar
            try:
                return {func.__name__: func(s2) - func(s1)}
            except:
                return {func.__name__: np.nan}
            
        elif isinstance(param, dict):
            try:
                # Combiner function: return a list of (name, value)
                s1_result = dict(func(s1, [param]))
                s2_result = dict(func(s2, [param]))
                result = {}
                for k in s1_result.keys():
                    # print(s1_result[k], s2_result[k], k)
                    feat_name = f"{func.__name__}_{k}"
                    result[feat_name] = s2_result[k] - s1_result[k]
                return result
            except TypeError:
                # Simple function with multiple kwargs
                val = func(s2, **param) - func(s1, **param)
                feat_name = f"{func.__name__}_{param_to_str(param)}"
                return {feat_name: val}
            except Exception as e:
                # print(e)
                feat_name = f"{func.__name__}_{param_to_str(param)}"
                return {feat_name: np.nan}
            
        else:
            # Simple function with parameter
            try:
                feat_name = f"{func.__name__}_{param_to_str(param)}"
                return {feat_name: func(s2, param) - func(s1, param)}
            except:
                return {feat_name: np.nan}

    for func, params in funcs.items():
        if params is None:
            feats.update(cal_func_diff_as_feature(func))
        else:
            for param in params:
                feats.update(cal_func_diff_as_feature(func, param))

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}



# --- 特征管理核心逻辑 ---
def _get_latest_feature_file() -> Path | None:
    """查找并返回最新的特征文件路径"""
    feature_files = list(config.FEATURE_DIR.glob('features_*.parquet'))
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
    results = Parallel(n_jobs=-1)(
        delayed(lambda df_id, id_val: {**{'id': id_val}, **func(df_id)})(X_df.loc[id_val], id_val)
        for id_val in tqdm(all_ids, desc=f"Running {func.__name__}")
    )
    return pd.DataFrame(results).set_index('id')

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

def generate_features(X_df: pd.DataFrame, funcs_to_run: list = None, base_feature_file: str = None):
    """
    生成指定的特征，或者如果未指定，则生成所有已注册的特征。
    可以基于一个现有的特征文件进行增量更新。

    Args:
        X_df (pd.DataFrame): 包含 `series_id` 和 `time_step` 的输入数据。
        funcs_to_run (list, optional): 要运行的特征函数名称列表。
            如果为 None，则运行所有在 `FEATURE_REGISTRY` 中注册的、且不在 `EXPERIMENTAL_FEATURES` 中的函数。
        base_feature_file (str, optional): 基础特征文件名。如果提供，
            将加载此文件并在此基础上添加或更新特征。否则，将创建一个新的特征集。
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

    # 2. 逐个生成新特征并更新
    loaded_features = feature_df.columns.tolist()
    initial_feature_count = len(feature_df.columns)

    for func_name in funcs_to_run:
        logger.info(f"--- 开始生成特征: {func_name} ---")
        start_time = time.time()
        
        feature_info = FEATURE_REGISTRY[func_name]
        func = feature_info['func']
        is_parallelizable = feature_info['parallelizable']
        
        if is_parallelizable:
            new_features_df = _apply_feature_func_parallel(func, X_df)
        else:
            logger.info(f"函数 '{func_name}' 不可并行化，将顺序执行。")
            new_features_df = _apply_feature_func_sequential(func, X_df)
        
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
