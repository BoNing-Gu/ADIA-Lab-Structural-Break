from pathlib import Path

# --- Dirs ---
# 定义项目根目录
ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent

# --- Data & Feature Dirs ---
DATA_DIR = PROJECT_ROOT / 'data'
FEATURE_DIR = PROJECT_ROOT / 'feature_dfs'
FIGS_DIR = PROJECT_ROOT / 'figs'
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
N_JOBS = 72
SEED = 42

# --- Data Enhancement ---
# 数据增强配置，指定要加载的增强数据ID列表
# 如果为'0'，则只使用原始数据
ENHANCEMENT_IDS = ["0"] 

# --- Model ---
MODEL = 'LGB'  # 'LGB', 'CAT', or 'XGB'
LGBM_PARAMS = {
    # --- 基础设定 ---
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 3600, 
    'learning_rate': 0.005,
    'num_leaves': 29,
    'random_state': 42,
    'n_jobs': N_JOBS,

    # --- 正则化和采样 ---
    'reg_alpha': 3,            # L1 正则化
    'reg_lambda': 3,           # L2 正则化
    # 'min_child_samples': 50,   # 叶子节点样本量
    'colsample_bytree': 0.8,   # 构建树时对特征的列采样率
    'subsample': 0.8,          # 训练样本的采样率
}
CAT_PARAMS = {
    # --- 基础设定 ---
    'bootstrap_type': 'Bernoulli',
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    # 'task_type': 'GPU',
    'iterations': 3600, 
    'learning_rate': 0.005,
    'depth': 7,
    'random_seed': 114514,
    'thread_count': N_JOBS,
    
    # --- 正则化和采样 ---
    'subsample': 0.8,
    # 'rsm': 0.7,
    'l2_leaf_reg': 3,
}
XGB_PARAMS = {
    # --- 基础设定 ---
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    # 'device': 'cuda', 
    'n_estimators': 3600,
    'learning_rate': 0.005,
    'max_leaves': 29,
    'random_state': 2025,
    'n_jobs': N_JOBS,
    'verbosity': 0, 
    
    # --- 正则化和采样 ---
    'reg_alpha': 3,          # L1 正则化
    'reg_lambda': 3,         # L2 正则化
    'colsample_bytree': 0.8, # 构建树时对特征的列采样率
    'subsample': 0.8,        # 训练样本的采样率
}

# --- Early Stopping ---
# 设置为 >0 启用早停；设置为 0 禁用早停
EARLY_STOPPING_ROUNDS = 0

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

]

REMAIN_FEATURES = [
    'sqmul_RAW_1_stats_mean_whole_RAW_3_detrend_volatility_normalized_whole',
    'mul_RAW_3_detrend_volatility_normalized_whole_RAW_1_stats_std_whole',
    'mul_RAW_1_stats_cv_whole_RAW_1_stats_std_whole',
    'sub_RAW_8_ratio_value_number_to_time_series_length_ratio_to_whole_right_CUMSUM_4_autocorr_lag1_whole',
    'div_RAW_8_energy_ratio_by_chunks_num_segments_10_segment_focus_9_left_RAW_8_agg_linear_trend_attr_intercept_chunk_len_10_f_agg_max_ratio_to_whole_right',
    'div_RAW_2_ad_stat_RAW_8_percentage_of_reoccurring_values_to_all_values_contribution_left',
    'div_RAW_8_percentage_of_reoccurring_datapoints_to_all_datapoints_whole_RAW_8_percentage_of_reoccurring_values_to_all_values_diff',
    'add_RAW_8_index_mass_quantile_q_0_8_ratio_to_whole_left_RAW_8_index_mass_quantile_q_0_1_right',
    'mul_CUMSUM_2_ad_pvalue_DIFF_8_change_quantiles_f_agg_var_isabs_True_qh_0_4_ql_0_2_whole',
    'div_RAW_8_ratio_value_number_to_time_series_length_whole_CUMSUM_2_ad_pvalue',
    'div_RAW_1_stats_median_right_CUMSUM_1_stats_max_diff',
    'mul_RAW_2_bartlett_stat_DIFF_2_jb_pvalue_left',
    'sqmul_CUMSUM_1_stats_max_ratio_to_whole_left_RAW_8_percentage_of_reoccurring_values_to_all_values_ratio_to_whole_left',
    'mul_RAW_8_ratio_beyond_r_sigma_3_left_CUMSUM_1_stats_theil_sen_slope_whole',
    'mul_RAW_8_ratio_beyond_r_sigma_3_left_CUMSUM_1_stats_max_diff',
    'div_RAW_8_agg_linear_trend_attr_intercept_chunk_len_10_f_agg_max_ratio_to_whole_right_RAW_8_energy_ratio_by_chunks_num_segments_10_segment_focus_9_left',
    'div_RAW_7_sample_entropy_left_DIFF_2_jb_pvalue_left',
    'mul_RAW_8_agg_linear_trend_attr_slope_chunk_len_10_f_agg_mean_contribution_left_CUMSUM_1_stats_max_diff',
    'sqmul_RAW_8_index_mass_quantile_q_0_1_right_RAW_8_quantile_0_4_contribution_left',
    'sub_RAW_8_percentage_of_reoccurring_values_to_all_values_contribution_left_RAW_7_sample_entropy_left',
    'add_RAW_7_sample_entropy_left_CUMSUM_8_first_location_of_maximum_whole',
    'sub_DIFF_7_hjorth_complexity_contribution_right_RAW_7_higuchi_fd_contribution_right',
    'div_RAW_1_stats_median_whole_RAW_8_percentage_of_reoccurring_values_to_all_values_contribution_left',
    'div_RAW_8_percentage_of_reoccurring_values_to_all_values_contribution_left_RAW_8_percentage_of_reoccurring_datapoints_to_all_datapoints_whole',
    'div_RAW_8_percentage_of_reoccurring_values_to_all_values_contribution_left_RAW_8_percentage_of_reoccurring_values_to_all_values_ratio_to_whole_left',
    'sqmul_RAW_8_energy_ratio_by_chunks_num_segments_10_segment_focus_9_left_RAW_8_index_mass_quantile_q_0_1_right',
    'div_RAW_8_benford_correlation_whole_CUMSUM_2_ad_pvalue',
    'div_CUMSUM_8_ar_coefficient_coeff_2_k_10_ratio_RAW_1_stats_kurt_left',
    'sqmul_RAW_7_sample_entropy_left_RAW_8_count_above_0_whole',
    'mul_CUMSUM_2_wilcoxon_stat_RAW_8_change_quantiles_f_agg_var_isabs_True_qh_0_6_ql_0_4_ratio_to_whole_right',
    'add_RAW_8_ratio_beyond_r_sigma_3_left_DIFF_7_hjorth_complexity_ratio_to_whole_right',
    'sqmul_RAW_8_index_mass_quantile_q_0_1_right_RAW_8_index_mass_quantile_q_0_8_ratio_to_whole_left',
    'add_RAW_8_index_mass_quantile_q_0_1_right_RAW_8_index_mass_quantile_q_0_8_left',
    'sqmul_RAW_8_linear_trend_attr_pvalue_ratio_to_whole_left_CUMSUM_8_change_quantiles_f_agg_var_isabs_False_qh_0_8_ql_0_0_ratio_to_whole_right',
    'add_RAW_8_index_mass_quantile_q_0_1_right_RAW_8_energy_ratio_by_chunks_num_segments_10_segment_focus_9_left',
    'div_DIFF_8_change_quantiles_f_agg_var_isabs_False_qh_0_6_ql_0_4_whole_DIFF_8_change_quantiles_f_agg_var_isabs_True_qh_0_4_ql_0_2_whole',
    'div_RAW_8_change_quantiles_f_agg_var_isabs_False_qh_0_8_ql_0_2_diff_CUMSUM_9_ar_residuals_s1_pred_mean',
    'div_RAW_1_stats_min_whole_CUMSUM_8_friedrich_coefficients_coeff_3_m_3_r_30_ratio_to_whole_left',
    'sub_CUMSUM_8_change_quantiles_f_agg_var_isabs_False_qh_0_8_ql_0_0_whole_CUMSUM_2_ks_pvalue',
    'add_DIFF_2_bartlett_pvalue_DIFF_8_ar_coefficient_coeff_2_k_10_left',
    'sub_RAW_1_stats_std_whole_RAW_8_ratio_value_number_to_time_series_length_whole',
    'mul_RAW_8_index_mass_quantile_q_0_8_ratio_to_whole_left_RAW_8_ratio_value_number_to_time_series_length_ratio_to_whole_left',
    'div_RAW_3_detrend_volatility_normalized_whole_RAW_8_percentage_of_reoccurring_values_to_all_values_contribution_left',
    'sqmul_RAW_2_ks_stat_RAW_8_percentage_of_reoccurring_values_to_all_values_ratio_to_whole_left',
    'mul_RAW_8_index_mass_quantile_q_0_1_right_RAW_10_rpt_cost_cosine_whole',
    'mul_RAW_10_rpt_cost_cosine_whole_RAW_1_stats_median_ratio',
    'div_RAW_2_bartlett_stat_CUMSUM_7_cond_entropy_whole',
    'div_RAW_1_stats_mean_whole_RAW_8_ratio_beyond_r_sigma_3_left',
    'div_RAW_8_change_quantiles_f_agg_var_isabs_True_qh_0_6_ql_0_2_whole_RAW_1_stats_min_whole',
    'CUMSUM_8_benford_correlation_left',
    'sqmul_CUMSUM_3_detrend_volatility_normalized_right_RAW_8_quantile_0_4_right',
    'div_RAW_8_percentage_of_reoccurring_values_to_all_values_contribution_left_CUMSUM_5_dominant_freq_ratio_to_whole_right',
    'DIFF_7_spectral_entropy_ratio_to_whole_left',
    'div_RAW_7_approx_entropy_left_CUMSUM_2_ad_pvalue',
    'div_RAW_7_sample_entropy_left_RAW_8_ratio_beyond_r_sigma_1_left',
    'div_RAW_7_sample_entropy_whole_RAW_8_ratio_beyond_r_sigma_1_5_whole',
    'sqmul_RAW_8_linear_trend_attr_pvalue_ratio_to_whole_left_DIFF_8_change_quantiles_f_agg_var_isabs_False_qh_1_0_ql_0_2_ratio',
    'mul_RAW_8_change_quantiles_f_agg_var_isabs_False_qh_0_6_ql_0_4_ratio_to_whole_right_CUMSUM_2_wilcoxon_stat',
    'sub_RAW_8_index_mass_quantile_q_0_8_ratio_to_whole_left_RAW_8_ratio_beyond_r_sigma_1_left',
    'mul_RAW_8_ratio_value_number_to_time_series_length_ratio_RAW_8_ratio_value_number_to_time_series_length_whole',
    'div_CUMSUM_1_stats_theil_sen_slope_whole_RAW_8_percentage_of_reoccurring_values_to_all_values_contribution_left',
    'mul_CUMSUM_2_ad_pvalue_RAW_8_change_quantiles_f_agg_var_isabs_False_qh_0_8_ql_0_0_right',
    'mul_CUMSUM_3_detrend_volatility_normalized_right_RAW_8_change_quantiles_f_agg_var_isabs_True_qh_0_6_ql_0_2_whole',
    'div_RAW_8_change_quantiles_f_agg_var_isabs_False_qh_0_6_ql_0_4_left_RAW_8_percentage_of_reoccurring_values_to_all_values_contribution_left',
    'div_RAW_8_energy_ratio_by_chunks_num_segments_10_segment_focus_9_left_DIFF_8_change_quantiles_f_agg_var_isabs_False_qh_1_0_ql_0_2_ratio',
    'div_DIFF_2_levene_pvalue_RAW_1_stats_median_right',
    'mul_RAW_8_ratio_value_number_to_time_series_length_diff_CUMSUM_3_trend_normalized_slope_whole',
    'sub_RAW_8_energy_ratio_by_chunks_num_segments_10_segment_focus_9_left_RAW_8_quantile_0_1_contribution_right',
    'sqmul_RAW_8_agg_linear_trend_attr_intercept_chunk_len_10_f_agg_max_ratio_to_whole_right_RAW_8_ratio_beyond_r_sigma_1_5_whole',
    'div_RAW_2_ks_stat_RAW_8_change_quantiles_f_agg_var_isabs_False_qh_0_8_ql_0_0_right',
    'add_RAW_7_sample_entropy_left_DIFF_2_bartlett_pvalue',
    'sqmul_RAW_8_quantile_0_4_whole_RAW_7_katz_fd_whole',
    'mul_RAW_8_agg_linear_trend_attr_slope_chunk_len_10_f_agg_mean_contribution_left_DIFF_2_jb_pvalue_left',
    'add_RAW_7_sample_entropy_left_RAW_2_bartlett_pvalue',
    'sqmul_RAW_8_ratio_value_number_to_time_series_length_ratio_to_whole_right_RAW_8_ratio_value_number_to_time_series_length_ratio_to_whole_left',
    'sqmul_RAW_2_shapiro_pvalue_whole_RAW_2_bartlett_stat',
    'sqmul_RAW_8_change_quantiles_f_agg_var_isabs_True_qh_0_6_ql_0_4_ratio_to_whole_right_CUMSUM_7_hjorth_complexity_whole',
    'sub_RAW_8_percentage_of_reoccurring_values_to_all_values_contribution_left_DIFF_7_sample_entropy_right',
    'sqmul_CUMSUM_7_cond_entropy_whole_RAW_1_stats_kurt_left',
    'add_RAW_8_count_above_0_whole_CUMSUM_8_change_quantiles_f_agg_mean_isabs_True_qh_1_0_ql_0_4_left',
    'mul_DIFF_2_levene_pvalue_CUMSUM_3_detrend_volatility_normalized_right',
    'div_RAW_1_stats_kurt_left_RAW_1_stats_std_whole',
    'add_RAW_8_index_mass_quantile_q_0_1_right_RAW_8_ratio_value_number_to_time_series_length_whole',
    'div_RAW_8_quantile_0_4_right_RAW_8_change_quantiles_f_agg_var_isabs_False_qh_0_8_ql_0_4_whole',
    'mul_RAW_8_energy_ratio_by_chunks_num_segments_10_segment_focus_9_left_RAW_8_quantile_0_4_contribution_left',
    'DIFF_8_change_quantiles_f_agg_var_isabs_False_qh_0_8_ql_0_2_ratio_to_whole_left',
    'sqmul_RAW_8_ratio_value_number_to_time_series_length_ratio_to_whole_left_DIFF_8_benford_correlation_whole',
    'div_RAW_8_agg_linear_trend_attr_intercept_chunk_len_10_f_agg_max_ratio_to_whole_right_RAW_8_agg_linear_trend_attr_slope_chunk_len_10_f_agg_mean_contribution_left',
    'mul_RAW_8_benford_correlation_whole_RAW_8_count_above_0_whole',
    'div_RAW_8_ratio_value_number_to_time_series_length_whole_RAW_7_katz_fd_whole',
    'sub_RAW_8_percentage_of_reoccurring_values_to_all_values_ratio_to_whole_left_RAW_8_percentage_of_reoccurring_datapoints_to_all_datapoints_ratio_to_whole_left',
    'RAW_8_ratio_beyond_r_sigma_0_5_contribution_left',
    'sqmul_RAW_8_change_quantiles_f_agg_var_isabs_True_qh_0_6_ql_0_4_contribution_left_CUMSUM_7_katz_fd_whole',
    'sqmul_RAW_8_friedrich_coefficients_coeff_3_m_3_r_30_ratio_CUMSUM_8_friedrich_coefficients_coeff_3_m_3_r_30_ratio_to_whole_left',
    'sub_RAW_1_stats_min_whole_RAW_7_approx_entropy_whole',
    'mul_RAW_8_index_mass_quantile_q_0_1_right_RAW_8_energy_ratio_by_chunks_num_segments_10_segment_focus_9_left',
    'div_RAW_2_bartlett_pvalue_CUMSUM_2_kpss_pvalue_ratio_to_whole_left',
    'add_DIFF_2_bartlett_pvalue_RAW_7_sample_entropy_whole',
    'sub_RAW_8_ratio_value_number_to_time_series_length_ratio_to_whole_right_RAW_2_shapiro_pvalue_whole',
    'CUMSUM_2_adf_pvalue_ratio_to_whole_right',
    'div_DIFF_2_bartlett_pvalue_RAW_8_agg_linear_trend_attr_rvalue_chunk_len_50_f_agg_max_ratio_to_whole_left',
    'div_RAW_7_sample_entropy_left_CUMSUM_2_ad_pvalue',
    'div_CUMSUM_2_adf_stat_diff_RAW_8_friedrich_coefficients_coeff_3_m_3_r_30_ratio',
    'div_CUMSUM_3_detrend_volatility_normalized_right_CUMSUM_1_stats_range_right',
    'div_RAW_8_percentage_of_reoccurring_values_to_all_values_ratio_to_whole_left_RAW_8_percentage_of_reoccurring_values_to_all_values_contribution_left',
    'add_DIFF_7_hjorth_complexity_contribution_right_RAW_1_stats_mean_right',
    'div_RAW_1_stats_median_right_RAW_8_percentage_of_reoccurring_values_to_all_values_ratio',
]