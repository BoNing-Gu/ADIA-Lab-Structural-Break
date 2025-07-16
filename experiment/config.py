import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

# --- Dirs ---
# 定义项目根目录
ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent

# --- Data & Feature Dirs ---
DATA_DIR = PROJECT_ROOT / 'data'
FEATURE_DIR = PROJECT_ROOT / 'feature_dfs'
FEATURE_BACKUP_DIR = FEATURE_DIR / 'backups'

# --- Output & Log Dirs ---
OUTPUT_DIR = ROOT_DIR / 'output'
LOG_DIR = ROOT_DIR / 'logs'

# 新增：为不同类型的日志创建专门的子目录
FEATURE_LOG_DIR = LOG_DIR / 'feature_logs'
TRAINING_LOG_DIR = LOG_DIR / 'training_logs'

# --- Files ---
TRAIN_X_FILE = DATA_DIR / 'X_train.parquet'
TRAIN_Y_FILE = DATA_DIR / 'y_train.parquet'
# 不再有固定的特征文件

# --- Model ---
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'random_state': 42,
    'n_estimators': 4000, 
    'learning_rate': 0.005,
    'num_leaves': 31,
    'n_jobs': -1,

    # --- 正则化和采样 ---
    'reg_alpha': 0.5,          # L1 正则化
    'reg_lambda': 0.5,         # L2 正则化
    'colsample_bytree': 0.8,   # 构建树时对特征的列采样率
    'subsample': 0.8,          # 训练样本的采样率
}

# --- CV ---
CV_PARAMS = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42
} 

# --- Features ---
# 在这里定义不希望在 "一键生成所有特征" 时运行的函数名称
# 如果要运行这些特征，需要在命令行中通过 --funcs 参数明确指定
# 例如: python -m experiment.main gen-feats --funcs ar_model_features
EXPERIMENTAL_FEATURES = [
    "wavelet_features",
] 

# --- Drop Features ---
DROP_FEATURES = [
    'RAW_12_rpt_cost_rank_whole', 'RAW_8_approx_entropy_1', 'RAW_1_stats_theil_sen_slope_ratio', 'RAW_8_spectral_entropy_1', 'RAW_10_change_quantiles_f_agg_var_isabs_False_qh_0_4_ql_0_2', 'RAW_1_stats_skew_left', 'RAW_2_kpss_pvalue_right', 'RAW_12_rpt_cost_mahalanobis_left', 'RAW_12_rpt_cost_normal_whole', 'RAW_9_petrosian_fd_diff', 'RAW_1_stats_max_right', 'RAW_1_stats_mean_diff', 'RAW_10_fft_coefficient_attr_imag_coeff_1', 'RAW_10_change_quantiles_f_agg_var_isabs_True_qh_1_0_ql_0_4', 'RAW_5_dominant_freq_diff', 'RAW_10_change_quantiles_f_agg_var_isabs_True_qh_0_8_ql_0_4', 'RAW_11_param_3_diff', 'RAW_1_stats_skew_whole', 'RAW_1_stats_skew_diff', 'RAW_1_stats_kurt_diff', 'RAW_2_mannwhitney_stat', 'RAW_2_adf_pvalue_diff', 'RAW_1_stats_cv_ratio', 'RAW_10_change_quantiles_f_agg_var_isabs_True_qh_0_8_ql_0_6', 'RAW_8_shannon_entropy_1', 'RAW_10_agg_linear_trend_attr_slope_chunk_len_5_f_agg_mean_', 'RAW_2_wilcoxon_pvalue', 'RAW_10_agg_linear_trend_attr_rvalue_chunk_len_5_f_agg_mean_', 'RAW_8_svd_entropy_0', 'RAW_2_mannwhitney_pvalue', 'RAW_1_stats_theil_sen_slope_right', 'RAW_1_stats_range_whole', 'RAW_4_zero_cross_diff', 'RAW_1_stats_skew_ratio', 'RAW_8_svd_entropy_1', 'RAW_8_lziv_complexity_0', 'RAW_10_change_quantiles_f_agg_var_isabs_True_qh_1_0_ql_0_2', 'RAW_1_stats_max_ratio', 'RAW_10_fft_coefficient_attr_imag_coeff_2', 'RAW_12_rpt_cost_rank_right', 'RAW_11_param_4_diff', 'RAW_1_stats_theil_sen_slope_whole'
]
