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