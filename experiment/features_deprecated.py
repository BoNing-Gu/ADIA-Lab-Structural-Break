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