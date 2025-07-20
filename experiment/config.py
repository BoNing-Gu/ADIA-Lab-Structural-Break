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

# --- Feature Engineer ---
N_JOBS = 12
SEED = 42

# --- Model ---
LGBM_PARAMS = {
    # --- 基础设定 ---
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 4000, 
    'learning_rate': 0.005,
    'num_leaves': 29,
    'random_state': SEED,
    'n_jobs': N_JOBS,

    # --- 正则化和采样 ---
    'reg_alpha': 3,          # L1 正则化
    'reg_lambda': 3,         # L2 正则化
    'colsample_bytree': 0.8,   # 构建树时对特征的列采样率
    'subsample': 0.8,          # 训练样本的采样率
}

# --- CV ---
CV_PARAMS = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': SEED
} 

# --- Exclude Features ---
# 在这里定义不希望在 "一键生成所有特征" 时运行的函数名称
# 如果要运行这些特征，需要在命令行中通过 --funcs 参数明确指定
# 例如: python -m experiment.main gen-feats --funcs ar_model_features
EXPERIMENTAL_FEATURES = [
    "chow_test_features",
] 

# --- Top Features ---
TOP_FEATURES = [
    'RAW_1_stats_cv_whole',
    'RAW_10_percentage_of_reoccurring_values_to_all_values_contribution_left',
    'RAW_10_percentage_of_reoccurring_datapoints_to_all_datapoints_contribution_left',
    'RAW_1_stats_std_whole',
    'RAW_8_sample_entropy_left',
    'RAW_2_ks_stat',
    'RAW_10_quantile_0_4_whole',
    'RAW_10_fft_coefficient_attr_imag_coeff_1_ratio_to_whole_left',
    'RAW_2_bartlett_pvalue',
    'RAW_10_percentage_of_reoccurring_values_to_all_values_ratio_to_whole_left',
]

# --- Remain Features ---
REMAIN_FEATURES = [
    'RAW_1_stats_cv_whole_mul_RAW_1_stats_std_whole',
    'RAW_1_stats_cv_whole',
    'RAW_8_sample_entropy_left',
    'RAW_10_percentage_of_reoccurring_datapoints_to_all_datapoints_contribution_left',
    'RAW_10_percentage_of_reoccurring_values_to_all_values_contribution_left',
    'RAW_10_quantile_0_6_ratio_to_whole_left',
    'RAW_10_agg_linear_trend_attr_rvalue_chunk_len_50_f_agg_max_ratio_to_whole_left',
    'RAW_1_stats_mean_whole',
    'RAW_10_fft_coefficient_attr_imag_coeff_1_whole',
    'RAW_8_sample_entropy_whole',
    'RAW_10_quantile_0_4_contribution_left',
    'RAW_4_autocorr_lag1_diff',
    'RAW_3_cumsum_max_ratio_to_whole_left',
    'RAW_2_ad_stat',
    'RAW_11_param_0_whole',
    'RAW_10_fft_coefficient_attr_imag_coeff_1_ratio_to_whole_right',
    'RAW_10_percentage_of_reoccurring_values_to_all_values_contribution_left_mul_RAW_10_percentage_of_reoccurring_datapoints_to_all_datapoints_contribution_left',
    'RAW_12_rpt_cost_cosine_whole',
    'RAW_11_param_3_right',
    'RAW_10_agg_linear_trend_attr_rvalue_chunk_len_50_f_agg_mean_diff',
    'RAW_1_stats_cv_whole_mul_RAW_10_percentage_of_reoccurring_datapoints_to_all_datapoints_contribution_left',
    'RAW_1_stats_median_ratio',
    'RAW_8_sample_entropy_left_mul_RAW_2_ks_stat',
    'RAW_2_ad_pvalue',
    'RAW_2_adf_icbest_left',
    'RAW_2_levene_stat',
    'RAW_8_perm_entropy_left',
    'RAW_10_linear_trend_attr_rvalue_ratio',
    'RAW_10_percentage_of_reoccurring_values_to_all_values_contribution_left_mul_RAW_10_percentage_of_reoccurring_values_to_all_values_ratio_to_whole_left',
    'RAW_1_stats_kurt_whole',
    'RAW_10_ratio_value_number_to_time_series_length_ratio_to_whole_left',
    'RAW_10_quantile_0_4_whole_mul_RAW_2_bartlett_pvalue',
    'RAW_10_change_quantiles_f_agg_var_isabs_True_qh_0_8_ql_0_4_contribution_left',
    'RAW_1_stats_median_whole',
    'RAW_2_levene_pvalue',
    'RAW_2_bartlett_pvalue',
    'RAW_10_ratio_value_number_to_time_series_length_whole',
    'RAW_10_agg_linear_trend_attr_rvalue_chunk_len_5_f_agg_max_left',
    'RAW_8_katz_fd_whole',
    'RAW_4_autocorr_lag1_ratio',
    'RAW_10_linear_trend_attr_intercept_ratio',
    'RAW_10_agg_linear_trend_attr_rvalue_chunk_len_50_f_agg_max_ratio',
    'RAW_8_svd_entropy_ratio_to_whole_left',
    'RAW_8_sample_entropy_left_mul_RAW_10_percentage_of_reoccurring_values_to_all_values_ratio_to_whole_left',
    'RAW_1_stats_min_diff',
    'RAW_1_stats_kurt_left',
    'RAW_10_change_quantiles_f_agg_var_isabs_False_qh_0_6_ql_0_4_ratio_to_whole_right',
    'RAW_8_sample_entropy_contribution_right',
    'RAW_2_ks_stat_mul_RAW_10_percentage_of_reoccurring_values_to_all_values_ratio_to_whole_left',
    'RAW_10_fft_coefficient_attr_imag_coeff_1_left',
    'RAW_10_quantile_0_4_right',
    'RAW_1_stats_cv_whole_mul_RAW_10_quantile_0_4_whole',
    'RAW_1_stats_mean_right',
    'RAW_10_change_quantiles_f_agg_var_isabs_True_qh_1_0_ql_0_4_contribution_left',
    'RAW_1_stats_cv_whole_mul_RAW_10_percentage_of_reoccurring_values_to_all_values_contribution_left',
    'RAW_10_agg_linear_trend_attr_rvalue_chunk_len_5_f_agg_max_diff',
    'RAW_10_agg_linear_trend_attr_intercept_chunk_len_10_f_agg_max_ratio_to_whole_right',
    'RAW_2_bartlett_stat',
    'RAW_2_ttest_pvalue',
    'RAW_8_perm_entropy_whole',
    'RAW_10_quantile_0_6_whole',
    'RAW_8_lziv_complexity_right',
    'RAW_2_ks_stat_mul_RAW_2_bartlett_pvalue',
    'RAW_10_percentage_of_reoccurring_values_to_all_values_right',
    'RAW_10_change_quantiles_f_agg_var_isabs_False_qh_0_8_ql_0_0_right',
    'RAW_8_approx_entropy_whole',
    'RAW_4_autocorr_lag1_whole',
    'RAW_10_ratio_beyond_r_sigma_1_5_diff',
    'RAW_10_percentage_of_reoccurring_values_to_all_values_ratio',
    'RAW_2_ks_stat_mul_RAW_10_fft_coefficient_attr_imag_coeff_1_ratio_to_whole_left',
    'RAW_8_sample_entropy_left_mul_RAW_10_fft_coefficient_attr_imag_coeff_1_ratio_to_whole_left',
    'RAW_10_fft_coefficient_attr_imag_coeff_1_ratio_to_whole_left_mul_RAW_2_bartlett_pvalue',
    'RAW_10_ratio_beyond_r_sigma_1_5_contribution_left',
    'RAW_2_ks_pvalue',
    'RAW_10_agg_linear_trend_attr_rvalue_chunk_len_5_f_agg_max_right',
    'RAW_5_max_power_whole',
    'RAW_11_param_5_diff',
    'RAW_1_stats_skew_right',
    'RAW_10_percentage_of_reoccurring_values_to_all_values_ratio_to_whole_left',
    'RAW_10_percentage_of_reoccurring_datapoints_to_all_datapoints_ratio_to_whole_left',
    'RAW_1_stats_std_whole_mul_RAW_2_ks_stat',
    'RAW_8_spectral_entropy_left',
    'RAW_1_stats_cv_whole_mul_RAW_8_sample_entropy_left',
    'RAW_10_agg_linear_trend_attr_intercept_chunk_len_50_f_agg_mean_ratio_to_whole_left',
    'RAW_11_param_0_ratio_to_whole_right',
    'RAW_10_agg_linear_trend_attr_rvalue_chunk_len_50_f_agg_max_contribution_left',
]