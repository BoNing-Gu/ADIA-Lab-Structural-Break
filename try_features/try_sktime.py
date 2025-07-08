import pandas as pd
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 需要安装的库
from sktime.detection.clasp import ClaSPSegmentation
from sktime.detection.igts import InformationGainSegmentation
from sktime.detection.bs import BinarySegmentation
from sktime.detection.hmm_learn import GaussianHMM 
from sktime.detection.clust import ClusterSegmenter
from sktime.detection.wclust import WindowSegmenter

from sktime.clustering.k_means import TimeSeriesKMeans
from skchange.change_detectors import MovingWindow, PELT, SeededBinarySegmentation
from skchange.anomaly_detectors import CAPA, CircularBinarySegmentation
from skchange.costs import MultivariateGaussianCost, MultivariateTCost, GaussianCost, LaplaceCost, L1Cost, L2Cost, LinearRegressionCost, LinearTrendCost, PoissonCost,


def advanced_cpd_segmentation(u: pd.DataFrame) -> Dict[str, float]:
    """使用高级分割方法进行特征工程，特别适合数据生成过程变化检测"""
    if not SKTIME_AVAILABLE:
        return {}
    
    signal = u['value'].to_numpy()
    s1 = u['value'][u['period'] == 0].reset_index(drop=True)
    s2 = u['value'][u['period'] == 1].reset_index(drop=True)
    feats = {}

    s1_length = len(s1)
    s2_length = len(s2)
    full_length = len(signal)
    boundary_idx = s1_length

    # 准备数据
    signal_df = pd.DataFrame({'value': signal})
    
    try:
        # 1. ClaSP Segmentation - 基于分类得分轮廓
        try:
            period_length = min(20, full_length // 10)  # 自适应周期长度
            clasp = ClaSPSegmentation(
                period_length=period_length,
                n_cps=5,  # 最多检测5个变化点
                exclusion_radius=5
            )
            clasp_result = clasp.fit_predict(signal_df)
            
            if len(clasp_result) > 0:
                detected_cp = clasp_result.iloc[0]
                feats['ClaSP_distance'] = detected_cp - boundary_idx
                feats['ClaSP_relative_distance'] = (detected_cp - boundary_idx) / full_length
                feats['ClaSP_cp_count'] = len(clasp_result)
                
                # 最近的变化点
                distances = np.abs(clasp_result.values - boundary_idx)
                feats['ClaSP_min_distance'] = np.min(distances)
                feats['ClaSP_closest_cp'] = clasp_result.iloc[np.argmin(distances)] - boundary_idx
            else:
                feats['ClaSP_distance'] = 0
                feats['ClaSP_relative_distance'] = 0
                feats['ClaSP_cp_count'] = 0
                feats['ClaSP_min_distance'] = 0
                feats['ClaSP_closest_cp'] = 0
                
        except Exception as e:
            feats['ClaSP_distance'] = 0
            feats['ClaSP_relative_distance'] = 0
            feats['ClaSP_cp_count'] = 0
            feats['ClaSP_min_distance'] = 0
            feats['ClaSP_closest_cp'] = 0

        # 2. Information Gain Segmentation
        try:
            igts = InformationGainSegmentation(
                k_max=min(10, full_length // 20),  # 最大分割数
                step=max(1, full_length // 100)   # 步长
            )
            igts_result = igts.fit_predict(signal_df)
            
            if len(igts_result) > 0:
                detected_cp = igts_result.iloc[0]
                feats['IGTS_distance'] = detected_cp - boundary_idx
                feats['IGTS_relative_distance'] = (detected_cp - boundary_idx) / full_length
                feats['IGTS_cp_count'] = len(igts_result)
                
                # 最近的变化点
                distances = np.abs(igts_result.values - boundary_idx)
                feats['IGTS_min_distance'] = np.min(distances)
            else:
                feats['IGTS_distance'] = 0
                feats['IGTS_relative_distance'] = 0
                feats['IGTS_cp_count'] = 0
                feats['IGTS_min_distance'] = 0
                
        except Exception as e:
            feats['IGTS_distance'] = 0
            feats['IGTS_relative_distance'] = 0
            feats['IGTS_cp_count'] = 0
            feats['IGTS_min_distance'] = 0

        # 3. Binary Segmentation with threshold
        try:
            # 计算自适应阈值
            signal_std = np.std(signal)
            threshold = 2 * signal_std
            
            binseg = BinarySegmentation(threshold=threshold)
            binseg_result = binseg.fit_predict(signal_df)
            
            if len(binseg_result) > 0:
                detected_cp = binseg_result.iloc[0]
                feats['BinSeg_distance'] = detected_cp - boundary_idx
                feats['BinSeg_relative_distance'] = (detected_cp - boundary_idx) / full_length
                feats['BinSeg_cp_count'] = len(binseg_result)
                
                # 最近的变化点
                distances = np.abs(binseg_result.values - boundary_idx)
                feats['BinSeg_min_distance'] = np.min(distances)
            else:
                feats['BinSeg_distance'] = 0
                feats['BinSeg_relative_distance'] = 0
                feats['BinSeg_cp_count'] = 0
                feats['BinSeg_min_distance'] = 0
                
        except Exception as e:
            feats['BinSeg_distance'] = 0
            feats['BinSeg_relative_distance'] = 0
            feats['BinSeg_cp_count'] = 0
            feats['BinSeg_min_distance'] = 0

        # 4. Gaussian HMM - 建模不同状态的随机过程
        try:
            n_states = min(3, max(2, full_length // 50))  # 自适应状态数
            hmm = GaussianHMM(n_components=n_states, covariance_type='full')
            hmm_result = hmm.fit_predict(signal_df)
            
            if len(hmm_result) > 0:
                # 找到状态变化点
                state_changes = []
                for i in range(1, len(hmm_result)):
                    if hmm_result.iloc[i] != hmm_result.iloc[i-1]:
                        state_changes.append(i)
                
                if state_changes:
                    first_change = state_changes[0]
                    feats['HMM_distance'] = first_change - boundary_idx
                    feats['HMM_relative_distance'] = (first_change - boundary_idx) / full_length
                    feats['HMM_state_changes'] = len(state_changes)
                    
                    # 最近的状态变化
                    distances = np.abs(np.array(state_changes) - boundary_idx)
                    feats['HMM_min_distance'] = np.min(distances)
                else:
                    feats['HMM_distance'] = 0
                    feats['HMM_relative_distance'] = 0
                    feats['HMM_state_changes'] = 0
                    feats['HMM_min_distance'] = 0
            else:
                feats['HMM_distance'] = 0
                feats['HMM_relative_distance'] = 0
                feats['HMM_state_changes'] = 0
                feats['HMM_min_distance'] = 0
                
        except Exception as e:
            feats['HMM_distance'] = 0
            feats['HMM_relative_distance'] = 0
            feats['HMM_state_changes'] = 0
            feats['HMM_min_distance'] = 0

    except Exception as e:
        # 如果整个过程失败，返回默认值
        default_features = [
            'ClaSP_distance', 'ClaSP_relative_distance', 'ClaSP_cp_count', 'ClaSP_min_distance', 'ClaSP_closest_cp',
            'IGTS_distance', 'IGTS_relative_distance', 'IGTS_cp_count', 'IGTS_min_distance',
            'BinSeg_distance', 'BinSeg_relative_distance', 'BinSeg_cp_count', 'BinSeg_min_distance',
            'HMM_distance', 'HMM_relative_distance', 'HMM_state_changes', 'HMM_min_distance'
        ]
        for feat in default_features:
            feats[feat] = 0

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}


def advanced_cpd_anomaly_detection(u: pd.DataFrame) -> Dict[str, float]:
    """使用高级异常检测方法，特别适合检测数据生成过程的异常段"""
    if not SKTIME_AVAILABLE:
        return {}
    
    signal = u['value'].to_numpy()
    s1 = u['value'][u['period'] == 0].reset_index(drop=True)
    s2 = u['value'][u['period'] == 1].reset_index(drop=True)
    feats = {}

    s1_length = len(s1)
    s2_length = len(s2)
    full_length = len(signal)
    boundary_idx = s1_length

    # 准备数据
    signal_df = pd.DataFrame({'value': signal})
    
    try:
        # 1. CAPA - 集合异常和点异常检测
        try:
            capa = CAPA(
                collective_saving=0.5,
                point_saving=0.5,
                min_segment_length=max(2, full_length // 50),
                max_segment_length=min(full_length // 2, 100)
            )
            capa_result = capa.fit_predict(signal_df)
            
            if len(capa_result) > 0:
                # CAPA返回的是异常段，需要提取变化点
                first_anomaly = capa_result.iloc[0]
                feats['CAPA_distance'] = first_anomaly - boundary_idx
                feats['CAPA_relative_distance'] = (first_anomaly - boundary_idx) / full_length
                feats['CAPA_anomaly_count'] = len(capa_result)
                
                # 最近的异常点
                distances = np.abs(capa_result.values - boundary_idx)
                feats['CAPA_min_distance'] = np.min(distances)
            else:
                feats['CAPA_distance'] = 0
                feats['CAPA_relative_distance'] = 0
                feats['CAPA_anomaly_count'] = 0
                feats['CAPA_min_distance'] = 0
                
        except Exception as e:
            feats['CAPA_distance'] = 0
            feats['CAPA_relative_distance'] = 0
            feats['CAPA_anomaly_count'] = 0
            feats['CAPA_min_distance'] = 0

        # 2. Circular Binary Segmentation
        try:
            cbs = CircularBinarySegmentation(
                cost=L2Cost(),
                min_segment_length=max(2, full_length // 50),
                max_segment_length=min(full_length // 2, 100)
            )
            cbs_result = cbs.fit_predict(signal_df)
            
            if len(cbs_result) > 0:
                first_segment = cbs_result.iloc[0]
                feats['CBS_distance'] = first_segment - boundary_idx
                feats['CBS_relative_distance'] = (first_segment - boundary_idx) / full_length
                feats['CBS_segment_count'] = len(cbs_result)
                
                # 最近的异常段
                distances = np.abs(cbs_result.values - boundary_idx)
                feats['CBS_min_distance'] = np.min(distances)
            else:
                feats['CBS_distance'] = 0
                feats['CBS_relative_distance'] = 0
                feats['CBS_segment_count'] = 0
                feats['CBS_min_distance'] = 0
                
        except Exception as e:
            feats['CBS_distance'] = 0
            feats['CBS_relative_distance'] = 0
            feats['CBS_segment_count'] = 0
            feats['CBS_min_distance'] = 0

    except Exception as e:
        # 如果整个过程失败，返回默认值
        default_features = [
            'CAPA_distance', 'CAPA_relative_distance', 'CAPA_anomaly_count', 'CAPA_min_distance',
            'CBS_distance', 'CBS_relative_distance', 'CBS_segment_count', 'CBS_min_distance'
        ]
        for feat in default_features:
            feats[feat] = 0

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}


def advanced_cpd_changepoint_detection(u: pd.DataFrame) -> Dict[str, float]:
    """使用高级变化点检测方法，包含更多的代价函数"""
    if not SKTIME_AVAILABLE:
        return {}
    
    signal = u['value'].to_numpy()
    s1 = u['value'][u['period'] == 0].reset_index(drop=True)
    s2 = u['value'][u['period'] == 1].reset_index(drop=True)
    feats = {}

    s1_length = len(s1)
    s2_length = len(s2)
    full_length = len(signal)
    boundary_idx = s1_length

    # 准备数据
    signal_df = pd.DataFrame({'value': signal})
    
    try:
        # 1. MovingWindow with NormalVarianceCost - 检测方差变化
        try:
            bandwidth = min(30, full_length // 6)
            mw_var = MovingWindow(
                change_score=NormalVarianceCost(),
                bandwidth=bandwidth,
                threshold_scale=2.0
            )
            mw_var_result = mw_var.fit_predict(signal_df)
            
            if len(mw_var_result) > 0:
                detected_cp = mw_var_result.iloc[0]
                feats['MW_Var_distance'] = detected_cp - boundary_idx
                feats['MW_Var_relative_distance'] = (detected_cp - boundary_idx) / full_length
                feats['MW_Var_cp_count'] = len(mw_var_result)
                
                # 最近的变化点
                distances = np.abs(mw_var_result.values - boundary_idx)
                feats['MW_Var_min_distance'] = np.min(distances)
            else:
                feats['MW_Var_distance'] = 0
                feats['MW_Var_relative_distance'] = 0
                feats['MW_Var_cp_count'] = 0
                feats['MW_Var_min_distance'] = 0
                
        except Exception as e:
            feats['MW_Var_distance'] = 0
            feats['MW_Var_relative_distance'] = 0
            feats['MW_Var_cp_count'] = 0
            feats['MW_Var_min_distance'] = 0

        # 2. PELT with NormalVarianceCost
        try:
            pelt_var = PELT(
                cost=NormalVarianceCost(),
                penalty_scale=2.0,
                min_segment_length=max(2, full_length // 50)
            )
            pelt_var_result = pelt_var.fit_predict(signal_df)
            
            if len(pelt_var_result) > 0:
                detected_cp = pelt_var_result.iloc[0]
                feats['PELT_Var_distance'] = detected_cp - boundary_idx
                feats['PELT_Var_relative_distance'] = (detected_cp - boundary_idx) / full_length
                feats['PELT_Var_cp_count'] = len(pelt_var_result)
                
                # 最近的变化点
                distances = np.abs(pelt_var_result.values - boundary_idx)
                feats['PELT_Var_min_distance'] = np.min(distances)
            else:
                feats['PELT_Var_distance'] = 0
                feats['PELT_Var_relative_distance'] = 0
                feats['PELT_Var_cp_count'] = 0
                feats['PELT_Var_min_distance'] = 0
                
        except Exception as e:
            feats['PELT_Var_distance'] = 0
            feats['PELT_Var_relative_distance'] = 0
            feats['PELT_Var_cp_count'] = 0
            feats['PELT_Var_min_distance'] = 0

        # 3. SeededBinarySegmentation - 更先进的二分分割
        try:
            sbs = SeededBinarySegmentation(
                cost=L2Cost(),
                min_segment_length=max(2, full_length // 50),
                max_segment_length=min(full_length // 2, 100)
            )
            sbs_result = sbs.fit_predict(signal_df)
            
            if len(sbs_result) > 0:
                detected_cp = sbs_result.iloc[0]
                feats['SBS_distance'] = detected_cp - boundary_idx
                feats['SBS_relative_distance'] = (detected_cp - boundary_idx) / full_length
                feats['SBS_cp_count'] = len(sbs_result)
                
                # 最近的变化点
                distances = np.abs(sbs_result.values - boundary_idx)
                feats['SBS_min_distance'] = np.min(distances)
            else:
                feats['SBS_distance'] = 0
                feats['SBS_relative_distance'] = 0
                feats['SBS_cp_count'] = 0
                feats['SBS_min_distance'] = 0
                
        except Exception as e:
            feats['SBS_distance'] = 0
            feats['SBS_relative_distance'] = 0
            feats['SBS_cp_count'] = 0
            feats['SBS_min_distance'] = 0

    except Exception as e:
        # 如果整个过程失败，返回默认值
        default_features = [
            'MW_Var_distance', 'MW_Var_relative_distance', 'MW_Var_cp_count', 'MW_Var_min_distance',
            'PELT_Var_distance', 'PELT_Var_relative_distance', 'PELT_Var_cp_count', 'PELT_Var_min_distance',
            'SBS_distance', 'SBS_relative_distance', 'SBS_cp_count', 'SBS_min_distance'
        ]
        for feat in default_features:
            feats[feat] = 0

    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}


def comprehensive_cpd_features(u: pd.DataFrame) -> Dict[str, float]:
    """综合所有高级CPD特征"""
    feats = {}
    
    # 获取所有特征
    seg_feats = advanced_cpd_segmentation(u)
    feats.update(seg_feats)
    
    anomaly_feats = advanced_cpd_anomaly_detection(u)
    feats.update(anomaly_feats)
    
    cp_feats = advanced_cpd_changepoint_detection(u)
    feats.update(cp_feats)
    
    # 计算方法间的一致性
    try:
        # 收集所有距离特征
        distance_features = [k for k in feats.keys() if 'distance' in k and 'relative' not in k and 'min' not in k]
        distances = [feats[k] for k in distance_features if feats[k] != 0]
        
        if len(distances) > 1:
            # 计算一致性指标
            feats['CPD_consensus_std'] = np.std(distances)
            feats['CPD_consensus_range'] = np.max(distances) - np.min(distances)
            feats['CPD_consensus_mean'] = np.mean(distances)
            feats['CPD_detection_rate'] = len(distances) / len(distance_features)
        else:
            feats['CPD_consensus_std'] = 0
            feats['CPD_consensus_range'] = 0
            feats['CPD_consensus_mean'] = 0
            feats['CPD_detection_rate'] = len(distances) / len(distance_features) if distance_features else 0
    except:
        feats['CPD_consensus_std'] = 0
        feats['CPD_consensus_range'] = 0
        feats['CPD_consensus_mean'] = 0
        feats['CPD_detection_rate'] = 0
    
    return {k: float(v) if not np.isnan(v) else 0 for k, v in feats.items()}


# 使用示例
if __name__ == "__main__":
    # 生成测试数据 - 模拟数据生成过程的变化
    np.random.seed(42)
    n1, n2 = 100, 100
    
    # 前半段：低波动率
    s1 = np.random.normal(0, 0.5, n1)
    # 后半段：高波动率
    s2 = np.random.normal(0, 2.0, n2)
    
    signal = np.concatenate([s1, s2])
    
    # 创建DataFrame
    u = pd.DataFrame({
        'value': signal,
        'period': [0] * n1 + [1] * n2
    })
    
    # 测试综合特征
    print("Testing comprehensive CPD features:")
    all_features = comprehensive_cpd_features(u)
    print(f"Total features: {len(all_features)}")
    
    # 显示一些关键特征
    for k, v in all_features.items():
        if any(x in k for x in ['ClaSP', 'IGTS', 'CAPA', 'consensus']):
            print(f"{k}: {v}")