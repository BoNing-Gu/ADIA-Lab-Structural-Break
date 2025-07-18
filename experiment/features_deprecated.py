# -*- coding: utf-8 -*-
"""
本文件用于存放已废弃或在实验中表现不佳的特征。

这里的特征函数不会被自动注册或运行。
如果需要重新启用某个特征，请将其代码移回 features.py 并重新添加 @register_feature 装饰器。
"""
import pandas as pd
import numpy as np

def mean_abs_diff_features(u: pd.DataFrame) -> dict:
    """
    计算基于平均绝对差分的特征。
    M = (1/(N-1)) * sum(|t_{i+1} - t_i|)
    """
    s1 = u['value'][u['period'] == 0]
    s2 = u['value'][u['period'] == 1]
    feats = {}

    m0 = s1.diff().abs().mean() if len(s1) > 1 else 0
    m1 = s2.diff().abs().mean() if len(s2) > 1 else 0

    feats['mean_abs_diff_0'] = m0
    feats['mean_abs_diff_1'] = m1
    feats['mean_abs_diff_diff'] = m1 - m0

    if m0 > 1e-6:
        feats['mean_abs_diff_rel_diff'] = (m1 - m0) / m0
    else:
        feats['mean_abs_diff_rel_diff'] = 0.0

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()} 


def more_distributional_stats(u: pd.DataFrame) -> dict:
    """
    比较两段时序数据的分布特征和统计性质
    """
    s1 = u['value'][u['period'] == 0].dropna()
    s2 = u['value'][u['period'] == 1].dropna()
    feats = {}
    
    # 确保数据不为空
    if len(s1) == 0 or len(s2) == 0:
        return {}
    
    # 1. 位置检验 (Location Tests)
    # t检验 (假设正态分布)
    try:
        ttest_stat, ttest_pvalue = scipy.stats.ttest_ind(s1, s2, equal_var=False)
        feats['ttest_pvalue'] = -ttest_pvalue if not np.isnan(ttest_pvalue) else 0
        feats['ttest_stat'] = ttest_stat if not np.isnan(ttest_stat) else 0
    except:
        feats['ttest_pvalue'] = 0
        feats['ttest_stat'] = 0
    
    # Mann-Whitney U检验 (非参数，不假设分布)
    try:
        mw_stat, mw_pvalue = scipy.stats.mannwhitneyu(s1, s2, alternative='two-sided')
        feats['mannwhitney_pvalue'] = -mw_pvalue if not np.isnan(mw_pvalue) else 0
        feats['mannwhitney_stat'] = mw_stat if not np.isnan(mw_stat) else 0
    except:
        feats['mannwhitney_pvalue'] = 0
        feats['mannwhitney_stat'] = 0
    
    # Wilcoxon秩和检验
    try:
        w_stat, w_pvalue = scipy.stats.ranksums(s1, s2)
        feats['wilcoxon_pvalue'] = -w_pvalue if not np.isnan(w_pvalue) else 0
        feats['wilcoxon_stat'] = w_stat if not np.isnan(w_stat) else 0
    except:
        feats['wilcoxon_pvalue'] = 0
        feats['wilcoxon_stat'] = 0
    
    # 2. 分布形状检验 (Distribution Shape Tests)
    # Kolmogorov-Smirnov检验
    try:
        ks_stat, ks_pvalue = scipy.stats.ks_2samp(s1, s2)
        feats['ks_pvalue'] = -ks_pvalue if not np.isnan(ks_pvalue) else 0
        feats['ks_stat'] = ks_stat if not np.isnan(ks_stat) else 0
    except:
        feats['ks_pvalue'] = 0
        feats['ks_stat'] = 0
    
    # Anderson-Darling检验
    try:
        ad_stat, ad_crit, ad_pvalue = scipy.stats.anderson_ksamp([s1, s2], method=stats.PermutationMethod(n_resamples=1000))
        feats['anderson_pvalue'] = -ad_pvalue if not np.isnan(ad_pvalue) else 0
        feats['anderson_stat'] = ad_stat if not np.isnan(ad_stat) else 0
    except:
        feats['anderson_pvalue'] = 0
        feats['anderson_stat'] = 0
    
    # 3. 方差齐性检验 (Variance Homogeneity Tests)
    # Levene检验
    try:
        levene_stat, levene_pvalue = scipy.stats.levene(s1, s2)
        feats['levene_pvalue'] = -levene_pvalue if not np.isnan(levene_pvalue) else 0
        feats['levene_stat'] = levene_stat if not np.isnan(levene_stat) else 0
    except:
        feats['levene_pvalue'] = 0
        feats['levene_stat'] = 0
    
    # Bartlett检验
    try:
        bartlett_stat, bartlett_pvalue = scipy.stats.bartlett(s1, s2)
        feats['bartlett_pvalue'] = -bartlett_pvalue if not np.isnan(bartlett_pvalue) else 0
        feats['bartlett_stat'] = bartlett_stat if not np.isnan(bartlett_stat) else 0
    except:
        feats['bartlett_pvalue'] = 0
        feats['bartlett_stat'] = 0
    
    # F检验方差比
    try:
        f_stat = np.var(s1, ddof=1) / np.var(s2, ddof=1)
        df1, df2 = len(s1) - 1, len(s2) - 1
        f_pvalue = 2 * min(scipy.stats.f.cdf(f_stat, df1, df2), 
                          1 - scipy.stats.f.cdf(f_stat, df1, df2))
        feats['f_test_pvalue'] = -f_pvalue if not np.isnan(f_pvalue) else 0
        feats['f_test_stat'] = f_stat if not np.isnan(f_stat) else 0
    except:
        feats['f_test_pvalue'] = 0
        feats['f_test_stat'] = 0
    
    # 4. 矩检验 (Moment Tests)
    # 偏度差异
    try:
        skew1 = scipy.stats.skew(s1)
        skew2 = scipy.stats.skew(s2)
        feats['skew_diff'] = skew1 - skew2 if not (np.isnan(skew1) or np.isnan(skew2)) else 0
    except:
        feats['skew_diff'] = 0
    
    # 峰度差异
    try:
        kurt1 = scipy.stats.kurtosis(s1)
        kurt2 = scipy.stats.kurtosis(s2)
        feats['kurtosis_diff'] = kurt1 - kurt2 if not (np.isnan(kurt1) or np.isnan(kurt2)) else 0
    except:
        feats['kurtosis_diff'] = 0
    
    # 5. 正态性检验 (Normality Tests)
    # Jarque-Bera检验差异
    try:
        jb1_stat, jb1_pvalue = scipy.stats.jarque_bera(s1)
        jb2_stat, jb2_pvalue = scipy.stats.jarque_bera(s2)
        feats['jb_pvalue_diff'] = jb1_pvalue - jb2_pvalue if not (np.isnan(jb1_pvalue) or np.isnan(jb2_pvalue)) else 0
    except:
        feats['jb_pvalue_diff'] = 0
    
    # Shapiro-Wilk检验差异
    try:
        if len(s1) <= 5000 and len(s2) <= 5000:  # Shapiro有样本大小限制
            sw1_stat, sw1_pvalue = scipy.stats.shapiro(s1)
            sw2_stat, sw2_pvalue = scipy.stats.shapiro(s2)
            feats['shapiro_pvalue_diff'] = sw1_pvalue - sw2_pvalue if not (np.isnan(sw1_pvalue) or np.isnan(sw2_pvalue)) else 0
        else:
            feats['shapiro_pvalue_diff'] = 0
    except:
        feats['shapiro_pvalue_diff'] = 0
    
    # 6. 分位数检验 (Quantile Tests)
    # 中位数差异
    try:
        median_diff = np.median(s1) - np.median(s2)
        feats['median_diff'] = median_diff if not np.isnan(median_diff) else 0
    except:
        feats['median_diff'] = 0
    
    # 四分位距差异
    try:
        iqr1 = np.percentile(s1, 75) - np.percentile(s1, 25)
        iqr2 = np.percentile(s2, 75) - np.percentile(s2, 25)
        feats['iqr_diff'] = iqr1 - iqr2 if not (np.isnan(iqr1) or np.isnan(iqr2)) else 0
    except:
        feats['iqr_diff'] = 0
    
    # 7. 时序特性检验 (Time Series Properties)
    # 自相关检验 (Ljung-Box)
    try:
        # 分别检验两个序列的自相关，输出1~10阶的pvalue差异
        if len(s1) > 10:
            lb1 = sm.stats.acorr_ljungbox(s1, lags=10, return_df=True)
            lb1_pvalues = lb1['lb_pvalue'].values
        else:
            lb1_pvalues = np.ones(10)
        if len(s2) > 10:
            lb2 = sm.stats.acorr_ljungbox(s2, lags=10, return_df=True)
            lb2_pvalues = lb2['lb_pvalue'].values
        else:
            lb2_pvalues = np.ones(10)
        # 构造10个特征，分别为每一阶的pvalue差异
        for i in range(10):
            p1 = lb1_pvalues[i] if i < len(lb1_pvalues) else 1.0
            p2 = lb2_pvalues[i] if i < len(lb2_pvalues) else 1.0
            feats[f'ljungbox_pvalue_diff_lag{i+1}'] = p1 - p2 if not (np.isnan(p1) or np.isnan(p2)) else 0
    except:
        for i in range(10):
            feats[f'ljungbox_pvalue_diff_lag{i+1}'] = 0
    
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

        feats['adf_pvalue_diff'] = f1['p'] - f2['p']
        feats['adf_stat_diff'] = f1['stat'] - f2['stat']
        feats['adf_lag_diff'] = f1['lag'] - f2['lag']
        feats['adf_icbest_diff'] = f1['ic'] - f2['ic']
        feats['adf_stat_relative_diff'] = (f1['stat'] - f1['crit_5pct']) - (f2['stat'] - f2['crit_5pct'])
        feats['adf_reject_flag_diff'] = f1['reject_5pct'] - f2['reject_5pct']
    except:
        feats['adf_pvalue_diff'] = 0
        feats['adf_stat_diff'] = 0
        feats['adf_lag_diff'] = 0
        feats['adf_icbest_diff'] = 0
        feats['adf_stat_relative_diff'] = 0
        feats['adf_reject_flag_diff'] = 0
    
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

        feats['kpss_pvalue_diff'] = k1['p'] - k2['p']
        feats['kpss_stat_diff'] = k1['stat'] - k2['stat']
        feats['kpss_lag_diff'] = k1['lag'] - k2['lag']
        feats['kpss_stat_relative_diff'] = (k1['stat'] - k1['crit_5pct']) - (k2['stat'] - k2['crit_5pct'])
        feats['kpss_reject_flag_diff'] = k1['reject_5pct'] - k2['reject_5pct']
    except:
        feats['kpss_pvalue_diff'] = 0
        feats['kpss_stat_diff'] = 0
        feats['kpss_lag_diff'] = 0
        feats['kpss_stat_relative_diff'] = 0
        feats['kpss_reject_flag_diff'] = 0
    
    # 8. 因果性检验 (Causality Tests)
    def extract_granger_features(x1, x2, max_lag=4):
        min_len = min(len(x1), len(x2))
        max_lag = min(max_lag, min_len // 4)
        if min_len <= 10 or max_lag < 1:
            return {'min_pval': 1.0, 'avg_pval': 1.0, 'best_lag': 0, 'pval_lag_diff': 0}
        
        data = pd.DataFrame({'x1': x1.iloc[-min_len:].values, 'x2': x2.iloc[:min_len].values})
        try:
            res = tsa.stattools.grangercausalitytests(data[['x2', 'x1']], maxlag=max_lag, verbose=False)
            # print(res)
            pvals = []
            for lag, (test_results, _) in res.items():
                # print(lag)
                # print(test_results)
                pval = test_results['ssr_ftest'][1]
                pvals.append((lag, pval))
            pvals_sorted = sorted(pvals, key=lambda x: x[1])
            min_pval = pvals_sorted[0][1]
            best_lag = pvals_sorted[0][0]
            return {
                'min_pval': min_pval,
                'best_lag': best_lag,
                'avg_pval': np.mean([p for _, p in pvals]),
                'pval_lag_diff': pvals[-1][1] - pvals[0][1],
            }
        except:
            return {'min_pval': 1.0, 'avg_pval': 1.0, 'best_lag': 0, 'pval_lag_diff': 0}

    try:
        fwd_stats = extract_granger_features(s1, s2)  # s1 → s2
        bwd_stats = extract_granger_features(s2, s1)  # s2 → s1

        feats['granger_fwd_min_pval'] = fwd_stats['min_pval']
        feats['granger_bwd_min_pval'] = bwd_stats['min_pval']
        feats['granger_pval_diff'] = fwd_stats['min_pval'] - bwd_stats['min_pval']
        feats['granger_avg_pval_diff'] = fwd_stats['avg_pval'] - bwd_stats['avg_pval']
        feats['granger_best_lag_diff'] = fwd_stats['best_lag'] - bwd_stats['best_lag']
        feats['granger_asym_causal'] = int((fwd_stats['min_pval'] < 0.05) and (bwd_stats['min_pval'] > 0.1))
    except:
        feats['granger_fwd_min_pval'] = 1.0
        feats['granger_bwd_min_pval'] = 1.0
        feats['granger_pval_diff'] = 0
        feats['granger_avg_pval_diff'] = 0
        feats['granger_best_lag_diff'] = 0
        feats['granger_asym_causal'] = 0
    
    # 9. 描述性统计差异 (Descriptive Statistics Differences)
    # 均值差异
    try:
        mean_diff = np.mean(s1) - np.mean(s2)
        feats['mean_diff'] = mean_diff if not np.isnan(mean_diff) else 0
    except:
        feats['mean_diff'] = 0
    
    # 标准差差异
    try:
        std_diff = np.std(s1, ddof=1) - np.std(s2, ddof=1)
        feats['std_diff'] = std_diff if not np.isnan(std_diff) else 0
    except:
        feats['std_diff'] = 0
    
    # 变异系数差异
    try:
        cv1 = np.std(s1, ddof=1) / np.mean(s1) if np.mean(s1) != 0 else 0
        cv2 = np.std(s2, ddof=1) / np.mean(s2) if np.mean(s2) != 0 else 0
        feats['cv_diff'] = cv1 - cv2 if not (np.isnan(cv1) or np.isnan(cv2)) else 0
    except:
        feats['cv_diff'] = 0
    
    # 10. 极值检验 (Extreme Value Tests)
    # 最大值差异
    try:
        max_diff = np.max(s1) - np.max(s2)
        feats['max_diff'] = max_diff if not np.isnan(max_diff) else 0
    except:
        feats['max_diff'] = 0
    
    # 最小值差异
    try:
        min_diff = np.min(s1) - np.min(s2)
        feats['min_diff'] = min_diff if not np.isnan(min_diff) else 0
    except:
        feats['min_diff'] = 0
    
    # 范围差异
    try:
        range1 = np.max(s1) - np.min(s1)
        range2 = np.max(s2) - np.min(s2)
        feats['range_diff'] = range1 - range2 if not (np.isnan(range1) or np.isnan(range2)) else 0
    except:
        feats['range_diff'] = 0
    
    # 清理NaN值
    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}


def wavelet_features(u: pd.DataFrame) -> dict:
    """
    使用小波变换提取时频域特征。
    对两个时期的数据分别进行多级小波分解，比较各层系数的统计特征。
    """
    import pywt
    s1 = u['value'][u['period'] == 0].to_numpy()
    s2 = u['value'][u['period'] == 1].to_numpy()
    feats = {}
    wavelet = 'db4'
    
    def get_wavelet_feats(series):
        if len(series) < pywt.Wavelet(wavelet).dec_len + 1:
            return {} # 数据太短无法分解
        
        # 自动决定分解层数
        max_level = pywt.dwt_max_level(len(series), pywt.Wavelet(wavelet).dec_len)
        levels = min(max_level, 4) # 限制最大层数为4，防止过分解
        if levels == 0:
            return {}
            
        coeffs = pywt.wavedec(series, wavelet, level=levels)
        
        w_feats = {}
        for i, (cA, *cDs) in enumerate(zip([coeffs[0]], coeffs[1:])):
            level = levels - i
            # 近似系数
            w_feats[f'wavelet_cA_energy_level{level}'] = np.sum(cA**2)
            w_feats[f'wavelet_cA_std_level{level}'] = np.std(cA)
            # 细节系数
            for j, cD in enumerate(cDs):
                detail_level = levels - j
                w_feats[f'wavelet_cD_energy_level{detail_level}'] = np.sum(cD**2)
                w_feats[f'wavelet_cD_std_level{detail_level}'] = np.std(cD)

        # 仅使用最后一层(最粗糙的近似)和所有细节系数
        last_cA = coeffs[0]
        all_details = np.concatenate(coeffs[1:])
        w_feats['wavelet_cA_last_energy'] = np.sum(last_cA**2)
        w_feats['wavelet_cA_last_std'] = np.std(last_cA)
        w_feats['wavelet_details_total_energy'] = np.sum(all_details**2)
        w_feats['wavelet_details_total_std'] = np.std(all_details)
        
        return w_feats

    s1_feats = get_wavelet_feats(s1)
    s2_feats = get_wavelet_feats(s2)

    all_keys = set(s1_feats.keys()) | set(s2_feats.keys())
    
    for key in all_keys:
        v1 = s1_feats.get(key, 0)
        v2 = s2_feats.get(key, 0)
        feats[f'{key}_diff'] = v2 - v1
        if abs(v1) > 1e-6:
             feats[f'{key}_ratio'] = v2 / v1
        else:
             feats[f'{key}_ratio'] = 1.0 if abs(v2) < 1e-6 else 1e6

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}
# # --- 11. 时间序列建模 ---
# @register_feature
# def ar_model_features(u: pd.DataFrame) -> dict:
#     import statsmodels.tsa.api as tsa
#     s1 = u['value'][u['period'] == 0].reset_index(drop=True)
#     s2 = u['value'][u['period'] == 1].reset_index(drop=True)
#     feats = {}

#     def fit_ar(s, lags=10, trend='ct'):
#         if len(s) <= lags + 1:
#             return None
#         try:
#             return tsa.AutoReg(s, lags=lags, trend=trend, old_names=False).fit()
#         except Exception:
#             return None

#     lags = 10
#     trend = 'ct'
#     model1_fwd = fit_ar(s1, lags, trend)
#     model2_fwd = fit_ar(s2, lags, trend)
#     model2_bwd = fit_ar(s2[::-1].reset_index(drop=True), lags, trend)

#     # --- 特征组1: model1_fwd 多步预测 s2 ---
#     try:
#         predictions_model1_fwd = model1_fwd.predict(start=len(s1), end=len(s1) + len(s2) - 1, dynamic=True)
#         residuals_model1_fwd = s2.to_numpy() - predictions_model1_fwd
#         feats['ar_resid_m1_fwd_mean'] = np.mean(residuals_model1_fwd)
#         feats['ar_resid_m1_fwd_std'] = np.std(residuals_model1_fwd)
#         feats['ar_resid_m1_fwd_skew'] = pd.Series(residuals_model1_fwd).skew()
#         feats['ar_resid_m1_fwd_kurt'] = pd.Series(residuals_model1_fwd).kurt()
#     except Exception:
#         feats.update({'ar_resid_m1_fwd_mean': 0, 'ar_resid_m1_fwd_std': 0, 'ar_resid_m1_fwd_skew': 0, 'ar_resid_m1_fwd_kurt': 0})

#     # --- 特征组2: model2_bwd 多步预测 s1 ---
#     try:
#         predictions_model2_bwd = model2_bwd.predict(start=len(s2), end=len(s2) + len(s1) - 1, dynamic=True)
#         residuals_model2_bwd = s1[::-1].to_numpy() - predictions_model2_bwd
#         feats['ar_resid_m2_bwd_mean'] = np.mean(residuals_model2_bwd)
#         feats['ar_resid_m2_bwd_std'] = np.std(residuals_model2_bwd)
#         feats['ar_resid_m2_bwd_skew'] = pd.Series(residuals_model2_bwd).skew()
#         feats['ar_resid_m2_bwd_kurt'] = pd.Series(residuals_model2_bwd).kurt()
#     except Exception:
#         feats.update({'ar_resid_m2_bwd_mean': 0, 'ar_resid_m2_bwd_std': 0, 'ar_resid_m2_bwd_skew': 0, 'ar_resid_m2_bwd_kurt': 0})

#     # --- 特征组2.5: model2_fwd 多步预测 s1 ---
#     try:
#         predictions_model2_fwd = model2_fwd.predict(start=len(s2), end=len(s2) + len(s1) - 1, dynamic=True)
#         residuals_model2_fwd = s1.to_numpy() - predictions_model2_fwd
#         feats['ar_resid_m2_fwd_mean'] = np.mean(residuals_model2_fwd)
#         feats['ar_resid_m2_fwd_std'] = np.std(residuals_model2_fwd)
#         feats['ar_resid_m2_fwd_skew'] = pd.Series(residuals_model2_fwd).skew()
#         feats['ar_resid_m2_fwd_kurt'] = pd.Series(residuals_model2_fwd).kurt()
#     except Exception:
#         feats.update({'ar_resid_m2_fwd_mean': 0, 'ar_resid_m2_fwd_std': 0, 'ar_resid_m2_fwd_skew': 0, 'ar_resid_m2_fwd_kurt': 0})

#     # --- 特征组3: 分别建模，比较差异 ---
#     if model1_fwd is not None and model2_fwd is not None:
#         feats['ar_resid_std_diff'] = model2_fwd.resid.std() - model1_fwd.resid.std()
#         feats['ar_aic_diff'] = model2_fwd.aic - model1_fwd.aic
#         feats['ar_const_diff'] = model2_fwd.params['const'] - model1_fwd.params['const']
#         feats['ar_trend_diff'] = model2_fwd.params['trend'] - model1_fwd.params['trend']
#         for i in range(1, lags + 1):
#             feats[f'ar_param_diff_{i}'] = model2_fwd.params[f'value.L{i}'] - model1_fwd.params[f'value.L{i}']
#     else:
#         feats['ar_resid_std_diff'] = 0.0
#         feats['ar_aic_diff'] = 0.0
#         feats['ar_const_diff'] = 0.0
#         feats['ar_trend_diff'] = 0.0
#         for i in range(1, lags + 1):
#             feats[f'ar_param_diff_{i}'] = 0.0

#     # --- 特征组4: model1 单步预测 s2 ---
#     try:
#         ar_lags = model1_fwd.model.ar_lags
#         max_lag = max(ar_lags)
#         history = s1[-max_lag:].tolist()
#         preds_manual = []

#         for t in range(len(s2)):
#             lagged_vals = history[-max_lag:]
#             pred = model1_fwd.params.get("const", 0.0)
#             if "trend" in model1_fwd.params.index:
#                 pred += model1_fwd.params["trend"] * (len(s1) + t)
#             for lag in ar_lags:
#                 pred += model1_fwd.params.get(f"value.L{lag}", 0.0) * lagged_vals[-lag]
#             preds_manual.append(pred)
#             history.append(s2[t])

#         preds_manual = np.array(preds_manual)
#         mse_manual = np.mean((preds_manual - s2.values[:len(preds_manual)]) ** 2)
#         feats["ar_autoreg_predict_mse"] = mse_manual

#     except Exception as e:
#         print(f"[WARN] Prediction error: {e}")
#         feats['ar_autoreg_predict_mse'] = 0.0

#     return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

# --- 9. ARIMA模型特征 ---
def arima_model_features_cpu(u: pd.DataFrame) -> dict:
    """
    基于ARIMA模型派生特征 (CPU版, 使用statsmodels)。
    1. 在 period 0 上训练模型，预测 period 1，计算残差统计量。
    2. 在 period 1 上训练模型，预测 period 0，计算残差统计量。
    3. 分别在 period 0 和 1 上训练模型，比较模型参数、残差和信息准则(AIC/BIC)。
    """
    from statsmodels.tsa.arima.model import ARIMA

    s1 = u['value'][u['period'] == 0].to_numpy()
    s2 = u['value'][u['period'] == 1].to_numpy()
    feats = {}
    order = (5, 1, 1) # (p, d, q)
    p, d, q = order
    min_len = p + d + 5 # A rough minimum length to fit ARIMA

    # --- 特征组1: 用 s1 训练，预测 s2 ---
    if len(s1) > min_len and len(s2) > 0:
        try:
            model1_fit = ARIMA(s1, order=order).fit()
            predictions = model1_fit.forecast(steps=len(s2))
            residuals = s2 - predictions
            feats['arima_residuals_s2_pred_mean'] = np.mean(residuals)
            feats['arima_residuals_s2_pred_std'] = np.std(residuals)
            feats['arima_residuals_s2_pred_skew'] = pd.Series(residuals).skew()
            feats['arima_residuals_s2_pred_kurt'] = pd.Series(residuals).kurt()
        except Exception:
            # 宽泛地捕获异常，防止因数值问题中断
            feats.update({'arima_residuals_s2_pred_mean': 0, 'arima_residuals_s2_pred_std': 0, 'arima_residuals_s2_pred_skew': 0, 'arima_residuals_s2_pred_kurt': 0})
    else:
        feats.update({'arima_residuals_s2_pred_mean': 0, 'arima_residuals_s2_pred_std': 0, 'arima_residuals_s2_pred_skew': 0, 'arima_residuals_s2_pred_kurt': 0})

    # --- 特征组2: 用 s2 训练，预测 s1 ---
    # This is less common but can capture reverse predictability changes.
    if len(s2) > min_len and len(s1) > 0:
        try:
            model2_fit = ARIMA(s2, order=order).fit()
            predictions_on_s1 = model2_fit.forecast(steps=len(s1))
            residuals_s1_pred = s1 - predictions_on_s1
            feats['arima_residuals_s1_pred_mean'] = np.mean(residuals_s1_pred)
            feats['arima_residuals_s1_pred_std'] = np.std(residuals_s1_pred)
            feats['arima_residuals_s1_pred_skew'] = pd.Series(residuals_s1_pred).skew()
            feats['arima_residuals_s1_pred_kurt'] = pd.Series(residuals_s1_pred).kurt()
        except Exception:
            feats.update({'arima_residuals_s1_pred_mean': 0, 'arima_residuals_s1_pred_std': 0, 'arima_residuals_s1_pred_skew': 0, 'arima_residuals_s1_pred_kurt': 0})
    else:
        feats.update({'arima_residuals_s1_pred_mean': 0, 'arima_residuals_s1_pred_std': 0, 'arima_residuals_s1_pred_skew': 0, 'arima_residuals_s1_pred_kurt': 0})

    # --- 特征组3: 分别建模，比较差异 ---
    s1_resid_std, s1_aic, s1_bic = np.nan, np.nan, np.nan
    s1_params = {}
    if len(s1) > min_len:
        try:
            fit1 = ARIMA(s1, order=order).fit()
            s1_resid_std = np.std(fit1.resid)
            s1_params = fit1.params.to_dict()
            s1_aic = fit1.aic
            s1_bic = fit1.bic
        except Exception:
            pass

    s2_resid_std, s2_aic, s2_bic = np.nan, np.nan, np.nan
    s2_params = {}
    if len(s2) > min_len:
        try:
            fit2 = ARIMA(s2, order=order).fit()
            s2_resid_std = np.std(fit2.resid)
            s2_params = fit2.params.to_dict()
            s2_aic = fit2.aic
            s2_bic = fit2.bic
        except Exception:
            pass
            
    feats['arima_resid_std_diff'] = (s2_resid_std - s1_resid_std) if not (np.isnan(s1_resid_std) or np.isnan(s2_resid_std)) else 0
    feats['arima_aic_diff'] = (s2_aic - s1_aic) if not (np.isnan(s1_aic) or np.isnan(s2_aic)) else 0
    feats['arima_bic_diff'] = (s2_bic - s1_bic) if not (np.isnan(s1_bic) or np.isnan(s2_bic)) else 0
    
    # 比较模型系数
    # Get all possible param names from both models
    all_param_names = sorted(list(set(s1_params.keys()) | set(s2_params.keys())))
    for name in all_param_names:
        v1 = s1_params.get(name, 0)
        v2 = s2_params.get(name, 0)
        # Clean name for feature
        clean_name = name.replace('.', '_')
        feats[f'arima_param_{clean_name}_diff'] = v2 - v1

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}

def arima_model_features_gpu(u: pd.DataFrame) -> dict:
    """
    基于ARIMA模型派生特征 (GPU加速版, 使用StatsForecast)。
    注意: StatsForecast 不计算 AIC/BIC，因此相关特征在此版本中被省略。
    """
    from statsforecast import StatsForecast
    from statsforecast.models import ARIMA

    s1_pd = u['value'][u['period'] == 0]
    s2_pd = u['value'][u['period'] == 1]
    feats = {}
    order = (5, 1, 1) # (p, d, q)
    p, d, q = order
    min_len = p + d + 5 # A rough minimum length to fit ARIMA

    df1_pd = pd.DataFrame({'unique_id': 's1', 'ds': np.arange(len(s1_pd)), 'y': s1_pd.values})
    df2_pd = pd.DataFrame({'unique_id': 's2', 'ds': np.arange(len(s2_pd)), 'y': s2_pd.values})
    
    df1 = cudf.from_pandas(df1_pd)
    df2 = cudf.from_pandas(df2_pd)
    
    # --- 特征组1: 用 s1 训练，预测 s2 ---
    if len(s1_pd) > min_len and len(s2_pd) > 0:
        try:
            sf1 = StatsForecast(models=[ARIMA(order=order, season_length=0)], freq=1)
            sf1.fit(df1)
            forecasts = sf1.predict(h=len(s2_pd))
            predictions = forecasts['ARIMA'].to_cupy().get()
            residuals = s2_pd.values - predictions
            feats['arima_residuals_s2_pred_mean'] = np.mean(residuals)
            feats['arima_residuals_s2_pred_std'] = np.std(residuals)
            feats['arima_residuals_s2_pred_skew'] = pd.Series(residuals).skew()
            feats['arima_residuals_s2_pred_kurt'] = pd.Series(residuals).kurt()
        except Exception:
            feats.update({'arima_residuals_s2_pred_mean': 0, 'arima_residuals_s2_pred_std': 0, 'arima_residuals_s2_pred_skew': 0, 'arima_residuals_s2_pred_kurt': 0})
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

@register_feature
def distance_features(u: pd.DataFrame) -> dict:
    """
    计算两个时间序列之间的DTW和EDR距离特征
    
    注意：
    - DTW距离要求等长序列，对不等长序列需要预处理
    - EDR距离可以处理不等长序列
    """
    from sktime.distances import dtw_distance, edr_distance
    
    s1 = u['value'][u['period'] == 0].to_numpy()
    s2 = u['value'][u['period'] == 1].to_numpy()
    feats = {}
    
    # # 1. DTW距离特征
    # try:
    #     # 不等长序列：分别用截断和插值方法处理
    #     # 方法1：截断到较短长度
    #     min_len = min(len(s1), len(s2))
    #     s1_truncated = s1[:min_len]
    #     s2_truncated = s2[:min_len]
        
    #     feats['dtw_distance_truncated'] = dtw_distance(s1_truncated, s2_truncated)
    #     feats['dtw_distance_truncated_normalized'] = feats['dtw_distance_truncated'] / min_len
        
    #     # 方法2：插值到相同长度
    #     target_length = max(len(s1), len(s2))  # 限制最大长度避免计算过慢
        
    #     def interpolate_series(series, target_len):
    #         """将序列插值到目标长度"""
    #         if len(series) == target_len:
    #             return series
    #         old_indices = np.linspace(0, len(series)-1, len(series))
    #         new_indices = np.linspace(0, len(series)-1, target_len)
    #         return np.interp(new_indices, old_indices, series)
        
    #     s1_interp = interpolate_series(s1, target_length)
    #     s2_interp = interpolate_series(s2, target_length)
        
    #     feats['dtw_distance_interpolated'] = dtw_distance(s1_interp, s2_interp)
    #     feats['dtw_distance_interpolated_normalized'] = feats['dtw_distance_interpolated'] / target_length
        
    #     # 带窗口约束的插值DTW
    #     window_ratio = 0.25
    #     feats['dtw_distance_interpolated_windowed'] = dtw_distance(
    #         s1_interp, s2_interp, window=window_ratio
    #     )
            
    # except Exception as e:
    #     print(f"DTW计算错误: {e}")
    #     # 为所有可能的DTW特征设置默认值
    #     dtw_features = [
    #         'dtw_distance', 'dtw_distance_windowed', 'dtw_distance_normalized',
    #         'dtw_distance_truncated', 'dtw_distance_truncated_normalized',
    #         'dtw_distance_interpolated', 'dtw_distance_interpolated_normalized',
    #         'dtw_distance_interpolated_windowed', 'dtw_distance_padded',
    #         'dtw_distance_padded_normalized'
    #     ]
    #     for feat in dtw_features:
    #         feats[feat] = 0
    
    # 2. EDR距离特征
    try:
        # EDR可以处理不等长序列，直接计算
        
        # 自适应epsilon：根据文档，默认是最大标准差的1/4
        combined_series = np.concatenate([s1, s2])
        std_s1 = np.std(s1) if len(s1) > 1 else 0
        std_s2 = np.std(s2) if len(s2) > 1 else 0
        max_std = max(std_s1, std_s2)
        epsilon_auto = max_std * 0.25 if max_std > 0 else 0.1
        
        feats['edr_distance'] = edr_distance(s1, s2, epsilon=epsilon_auto)
        
        # 使用不同的epsilon值
        epsilon_small = epsilon_auto * 0.5  # 更严格的匹配
        epsilon_large = epsilon_auto * 2.0  # 更宽松的匹配
        
        feats['edr_distance_strict'] = edr_distance(s1, s2, epsilon=epsilon_small)
        feats['edr_distance_loose'] = edr_distance(s1, s2, epsilon=epsilon_large)
        
        # 固定epsilon值（基于数据的整体统计）
        epsilon_fixed = np.std(combined_series) * 0.1
        feats['edr_distance_fixed_eps'] = edr_distance(s1, s2, epsilon=epsilon_fixed)
        
        # 记录使用的epsilon值用于分析
        feats['edr_epsilon_used'] = epsilon_auto
        
    except Exception as e:
        print(f"EDR计算错误: {e}")
        edr_features = [
            'edr_distance', 'edr_distance_strict', 'edr_distance_loose',
            'edr_distance_fixed_eps', 'edr_distance_windowed', 'edr_epsilon_used'
        ]
        for feat in edr_features:
            feats[feat] = 0

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}


# --- 13. 实验性稳健变异系数 ---
def _quartile_cv(s: pd.Series) -> float:
    """稳健变异系数 (Quartile Coefficient of Variation)."""
    if len(s) < 4:  # Need at least 4 points for quartiles
        return 0.0
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    if abs(q3 + q1) < 1e-6:
        return 0.0
    return (q3 - q1) / (q3 + q1)

def _log_cv(s: pd.Series) -> float:
    """对数变异系数 (Logarithmic Coefficient of Variation)."""
    s_pos = s[s > 0]
    if len(s_pos) < 2:
        return 0.0
    log_s = np.log(s_pos)
    if len(log_s) < 2:
        return 0.0
    log_std = np.std(log_s)
    return np.sqrt(np.exp(log_std**2) - 1)

def _mad_cv(s: pd.Series) -> float:
    """基于MAD的变异系数 (Median Absolute Deviation based CV)."""
    if len(s) < 2:
        return 0.0
    median_val = np.median(s)
    if abs(median_val) < 1e-6:
        return 0.0
    mad = np.median(np.abs(s - median_val))
    return mad / abs(median_val)

def _iqr_cv(s: pd.Series) -> float:
    """广义变异系数的实现 (Interquartile Range based CV)."""
    if len(s) < 4:
        return 0.0
    median_val = np.median(s)
    if abs(median_val) < 1e-6:
        return 0.0
    iqr = scipy.stats.iqr(s)
    return iqr / abs(median_val)


def experimental_robust_cv_features(u: pd.DataFrame) -> dict:
    """
    实验性特征：计算多种稳健的变异系数。
    - 稳健变异系数 (Quartile CV): (Q3-Q1)/(Q3+Q1)
    - 对数变异系数 (Log CV): for log-normal data
    - 基于MAD的变异系数 (MAD CV): MAD / |Median|
    - 广义变异系数 (IQR CV): IQR / |Median|
    """
    s1 = u['value'][u['period'] == 0]
    s2 = u['value'][u['period'] == 1]
    s_whole = u['value']
    feats = {}

    cv_funcs = {
        'robust_quartile_cv': _quartile_cv,
        'robust_log_cv': _log_cv,
        'robust_mad_cv': _mad_cv,
        'robust_iqr_cv': _iqr_cv,
    }

    for name, func in cv_funcs.items():
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

