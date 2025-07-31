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

# --- Data Enhancement ---
# 数据增强配置，指定要加载的增强数据ID列表
# 如果为'0'，则只使用原始数据
ENHANCEMENT_IDS = ["0"] 

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
    
] 

# --- Top Features ---
TOP_FEATURES = [
    'RAW_1_stats_cv_whole',
    'RAW_8_percentage_of_reoccurring_datapoints_to_all_datapoints_contribution_left',
    'RAW_8_percentage_of_reoccurring_values_to_all_values_contribution_left',
    'RAW_3_cumsum_linear_trend_pvalue_whole',
    'RAW_8_index_mass_quantile_q_0_8_ratio_to_whole_left',
    'RAW_7_sample_entropy_left',
    'RAW_1_stats_std_whole',
    'RAW_8_quantile_0_4_whole',
    'RAW_8_index_mass_quantile_q_0_1_right',
    'RAW_8_percentage_of_reoccurring_values_to_all_values_ratio_to_whole_left',
]

# --- Remain Features ---
REMAIN_FEATURES = [
    'mul_RAW_1_stats_cv_whole_RAW_1_stats_std_whole',
    'add_RAW_8_index_mass_quantile_q_0_8_ratio_to_whole_left_RAW_8_index_mass_quantile_q_0_1_right',
    'sub_RAW_8_percentage_of_reoccurring_values_to_all_values_contribution_left_RAW_7_sample_entropy_left',
    'sub_RAW_8_percentage_of_reoccurring_datapoints_to_all_datapoints_contribution_left_RAW_7_sample_entropy_left',
    'div_RAW_7_sample_entropy_left_RAW_3_cumsum_linear_trend_pvalue_whole',
    'RAW_2_bartlett_stat',
    'RAW_2_bartlett_pvalue',
    'div_RAW_1_stats_std_whole_RAW_1_stats_cv_whole',
    'sqmul_RAW_8_index_mass_quantile_q_0_1_right_RAW_8_index_mass_quantile_q_0_8_ratio_to_whole_left',
    'RAW_8_ratio_value_number_to_time_series_length_whole',
    'RAW_8_energy_ratio_by_chunks_num_segments_10_segment_focus_9_left',
    'mul_RAW_1_stats_cv_whole_RAW_8_change_quantiles_f_agg_var_isabs_False_qh_0_8_ql_0_0_whole',
    'mul_RAW_1_stats_cv_whole_RAW_2_ad_stat',
    'RAW_2_ks_stat',
    'sqmul_RAW_1_stats_cv_whole_RAW_1_stats_std_whole',
    'div_RAW_8_percentage_of_reoccurring_datapoints_to_all_datapoints_contribution_left_RAW_8_percentage_of_reoccurring_values_to_all_values_ratio_to_whole_left',
    'RAW_3_cumsum_max_ratio_to_whole_left',
    'RAW_1_stats_median_ratio',
    'RAW_2_adf_icbest_left',
    'RAW_8_quantile_0_4_ratio_to_whole_left',
    'RAW_8_fft_coefficient_attr_imag_coeff_1_ratio_to_whole_left',
    'sqmul_RAW_1_stats_std_whole_RAW_1_stats_cv_whole',
    'div_RAW_1_stats_cv_whole_RAW_8_percentage_of_reoccurring_datapoints_to_all_datapoints_contribution_left',
    'RAW_4_diff_var_contribution_left',
    'RAW_1_stats_kurt_whole',
    'div_RAW_8_quantile_0_4_whole_RAW_1_stats_std_whole',
    'mul_RAW_1_stats_cv_whole_RAW_1_stats_min_whole',
    'RAW_8_agg_linear_trend_attr_rvalue_chunk_len_50_f_agg_max_ratio_to_whole_left',
    'RAW_2_ad_stat',
    'RAW_8_benford_correlation_whole',
    'RAW_7_katz_fd_whole',
    'RAW_2_ad_pvalue',
    'sub_RAW_8_index_mass_quantile_q_0_8_ratio_to_whole_left_RAW_7_sample_entropy_left',
    'add_RAW_8_percentage_of_reoccurring_datapoints_to_all_datapoints_contribution_left_RAW_1_stats_std_whole',
    'RAW_7_perm_entropy_left',
    'RAW_8_first_location_of_maximum_left',
    'div_RAW_1_stats_std_whole_RAW_8_quantile_0_4_whole',
    'RAW_2_adf_stat_left',
    'mul_RAW_1_stats_cv_whole_RAW_8_agg_linear_trend_attr_slope_chunk_len_10_f_agg_mean_contribution_left',
    'div_RAW_8_percentage_of_reoccurring_values_to_all_values_contribution_left_RAW_8_percentage_of_reoccurring_values_to_all_values_ratio_to_whole_left',
    'RAW_2_ks_pvalue',
    'RAW_10_rpt_cost_cosine_whole',
    'RAW_8_ratio_beyond_r_sigma_1_left',
    'RAW_7_higuchi_fd_ratio_to_whole_right',
    'RAW_8_index_mass_quantile_q_0_8_left',
    'RAW_8_change_quantiles_f_agg_var_isabs_True_qh_1_0_ql_0_4_contribution_left',
    'RAW_8_linear_trend_attr_rvalue_ratio',
    'div_RAW_1_stats_cv_whole_RAW_8_quantile_0_4_whole',
    'RAW_8_index_mass_quantile_q_0_6_ratio_to_whole_left',
    'RAW_8_ratio_beyond_r_sigma_3_diff',
    'RAW_5_dominant_freq_right',
    'RAW_9_ar_param_5_diff',
    'RAW_8_quantile_0_1_ratio_to_whole_right',
    'RAW_8_ratio_beyond_r_sigma_0_5_diff',
    'RAW_8_quantile_0_6_right',
    'RAW_4_diff_var_contribution_right',
    'RAW_8_agg_linear_trend_attr_intercept_chunk_len_10_f_agg_max_ratio_to_whole_right',
    'mul_RAW_1_stats_cv_whole_RAW_8_ar_coefficient_coeff_2_k_10_left',
    'RAW_9_ar_param_3_right',
    'RAW_8_fft_coefficient_attr_imag_coeff_1_left',
    'RAW_8_change_quantiles_f_agg_var_isabs_False_qh_1_0_ql_0_2_ratio_to_whole_right',
    'RAW_8_benford_correlation_ratio_to_whole_left',
    'RAW_7_svd_entropy_whole',
    'RAW_8_ratio_beyond_r_sigma_1_ratio_to_whole_left',
    'div_RAW_1_stats_std_whole_RAW_3_cumsum_linear_trend_pvalue_whole',
    'RAW_3_cumsum_detrend_volatility_normalized_whole',
    'RAW_4_autocorr_lag1_diff',
    'div_RAW_8_percentage_of_reoccurring_datapoints_to_all_datapoints_contribution_left_RAW_8_percentage_of_reoccurring_values_to_all_values_contribution_left',
    'RAW_8_ratio_beyond_r_sigma_1_5_diff',
    'RAW_8_index_mass_quantile_q_0_6_right',
    'mul_RAW_1_stats_cv_whole_RAW_8_fft_coefficient_attr_imag_coeff_1_ratio_to_whole_right',
    'RAW_1_stats_median_right',
    'mul_RAW_1_stats_cv_whole_RAW_3_cumsum_max_ratio_to_whole_left',
    'RAW_9_ar_param_3_ratio_to_whole_left',
    'RAW_8_ratio_beyond_r_sigma_3_left',
    'RAW_8_agg_linear_trend_attr_rvalue_chunk_len_50_f_agg_mean_diff',
    'mul_RAW_1_stats_cv_whole_RAW_2_kpss_stat_whole',
    'RAW_8_fft_coefficient_attr_imag_coeff_1_ratio_to_whole_right',
    'RAW_2_ttest_pvalue',
    'RAW_8_change_quantiles_f_agg_var_isabs_False_qh_0_8_ql_0_0_right',
    'RAW_8_friedrich_coefficients_coeff_3_m_3_r_30_ratio',
    'RAW_4_autocorr_lag1_ratio',
    'RAW_1_stats_median_whole',
    'mul_RAW_1_stats_cv_whole_RAW_1_stats_min_diff',
    'RAW_8_sum_of_reoccurring_data_points_ratio_to_whole_left',
    'RAW_1_stats_kurt_left',
    'RAW_8_ratio_value_number_to_time_series_length_ratio_to_whole_left',
    'sqmul_RAW_1_stats_std_whole_RAW_7_sample_entropy_left',
    'mul_RAW_1_stats_cv_whole_RAW_8_fft_coefficient_attr_imag_coeff_3_left',
    'RAW_2_kpss_stat_whole',
    'mul_RAW_7_sample_entropy_left_RAW_8_percentage_of_reoccurring_values_to_all_values_ratio_to_whole_left',
    'RAW_3_cumsum_detrend_volatility_normalized_left',
    'RAW_8_quantile_0_4_right',
    'RAW_7_approx_entropy_left',
    'div_RAW_7_sample_entropy_left_RAW_8_index_mass_quantile_q_0_8_ratio_to_whole_left',
    'add_RAW_8_percentage_of_reoccurring_values_to_all_values_contribution_left_RAW_3_cumsum_linear_trend_pvalue_whole',
    'RAW_8_change_quantiles_f_agg_var_isabs_True_qh_0_6_ql_0_4_whole',
    'div_RAW_8_quantile_0_4_whole_RAW_3_cumsum_linear_trend_pvalue_whole',
    'sqmul_RAW_8_quantile_0_4_whole_RAW_7_sample_entropy_left',
    'add_RAW_8_percentage_of_reoccurring_datapoints_to_all_datapoints_contribution_left_RAW_8_quantile_0_4_whole',
    'RAW_3_cumsum_detrend_volatility_left',
    'mul_RAW_1_stats_cv_whole_RAW_8_agg_linear_trend_attr_rvalue_chunk_len_50_f_agg_mean_diff',
    'RAW_8_energy_ratio_by_chunks_num_segments_10_segment_focus_9_whole',
    'mul_RAW_1_stats_cv_whole_RAW_8_agg_linear_trend_attr_rvalue_chunk_len_50_f_agg_max_ratio_to_whole_left',
    'RAW_7_sample_entropy_ratio_to_whole_right',
    'mul_RAW_8_index_mass_quantile_q_0_8_ratio_to_whole_left_RAW_8_index_mass_quantile_q_0_1_right',
]