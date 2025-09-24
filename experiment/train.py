import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cat
import xgboost as xgb
from tqdm.auto import tqdm

# GPU数据处理支持
try:
    import cudf
    import cupy as cp
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    cudf = None
    cp = None

import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import logging
import json
import time
import joblib
from datetime import datetime

from . import config, features, data
from .model import DimReducer, NeighborFeatureExtractor

# logger 将由 main.py 在运行时注入
logger = None

def create_enhanced_cv_splits(feature_df, y_train, data_ids, cv_params):
    """
    创建增强数据的交叉验证分割策略。
    
    该函数确保：
    1. 只使用原始数据（索引0-10000）创建CV分割
    2. 验证集只包含原始数据
    3. 训练集包含原始数据及其对应的增强数据
    
    Args:
        feature_df: 包含所有数据（原始+增强）的特征DataFrame
        y_train: 包含所有数据（原始+增强）的标签Series
        data_ids: 数据增强ID列表，如["0", "1", "2"]
        cv_params: 交叉验证参数
    
    Returns:
        generator: 生成器，每次yield (train_idx, val_idx)
    """
    logger.info("创建增强数据交叉验证分割...")
    
    # 1. 识别原始数据索引（0-10000）
    original_indices = []
    enhanced_indices = {}  # {original_id: [enhanced_id1, enhanced_id2, ...]}
    
    for idx in feature_df.index:
        if idx <= 10000:  # 原始数据
            original_indices.append(idx)
            enhanced_indices[idx] = []
        else:  # 增强数据
            # 根据增强数据生成规律反推原始ID
            # new_id = int(func_id) * 1000000 + i * 100000 + int(original_id)
            original_id = idx % 100000  # 提取原始ID
            if original_id in enhanced_indices:
                enhanced_indices[original_id].append(idx)
            else:
                enhanced_indices[original_id] = [idx]
    
    original_indices = sorted(original_indices)
    logger.info(f"识别到 {len(original_indices)} 条原始数据")
    
    # 统计增强数据
    total_enhanced = sum(len(enhanced_list) for enhanced_list in enhanced_indices.values())
    logger.info(f"识别到 {total_enhanced} 条增强数据")
    
    # 2. 使用原始数据创建CV分割
    original_feature_df = feature_df.loc[original_indices]
    original_y_train = y_train.loc[original_indices]
    
    skf = StratifiedKFold(**cv_params)
    
    # 3. 为每个fold生成训练集和验证集索引
    for fold, (original_train_idx, original_val_idx) in enumerate(skf.split(original_feature_df, original_y_train)):
        # 获取原始数据的实际索引
        original_train_ids = [original_indices[i] for i in original_train_idx]
        original_val_ids = [original_indices[i] for i in original_val_idx]
        
        # 验证集只包含原始数据
        val_idx = original_val_ids
        
        # 训练集包含原始数据 + 对应的增强数据
        train_idx = original_train_ids.copy()
        
        # 为训练集中的每个原始数据添加对应的增强数据
        for original_id in original_train_ids:
            if original_id in enhanced_indices:
                train_idx.extend(enhanced_indices[original_id])
        
        # 转换为在feature_df中的位置索引
        train_positions = [feature_df.index.get_loc(idx) for idx in train_idx if idx in feature_df.index]
        val_positions = [feature_df.index.get_loc(idx) for idx in val_idx if idx in feature_df.index]
        
        logger.info(f"Fold {fold+1}: 训练集 {len(train_positions)} 条 (原始: {len(original_train_ids)}, 增强: {len(train_positions)-len(original_train_ids)}), 验证集 {len(val_positions)} 条 (仅原始数据)")
        
        yield train_positions, val_positions

def save_feature_importance(feature_importances, feature_names, output_dir):
    """将特征重要性降序保存为 tsv 文件。"""
    df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
    df = df.sort_values('importance', ascending=False)
    
    save_path = output_dir / 'feature_importance.tsv'
    df.to_csv(save_path, sep='\t', index=False)
    logger.info(f"特征重要性已保存到: {save_path}")

def save_permutation_importance(permutation_results, feature_names, output_dir):
    """将permutation importance保存为 tsv 文件。"""
    df = pd.DataFrame({
        'feature': feature_names,
        'permutation_importance_mean': permutation_results.mean(axis=1),
        'permutation_importance_std': permutation_results.std(axis=1)
    })
    df = df.sort_values('permutation_importance_mean', ascending=False)
    
    # 找出permutation_importance_mean小于threshold的特征名
    threshold = 0.0005
    negative_importance_features = df[df['permutation_importance_mean'] <= threshold]['feature'].tolist()
    logger.info(f"permutation_importance_mean小于{threshold}的特征有: {negative_importance_features}")

    save_path = output_dir / 'permutation_importance.tsv'
    df.to_csv(save_path, sep='\t', index=False)
    logger.info(f"Permutation importance已保存到: {save_path}")
    
    
def train_and_evaluate(feature_file_name: str, data_ids: list = ["0"], save_oof: bool = False, save_model: bool = False, perm_imp: bool = False):
    """
    加载指定的特征文件和标签，进行交叉验证，并根据参数选择性保存产出物。
    """
    start_time = time.time()
    logger.info("Starting training and evaluation pipeline...")
    # 根据实际选择的模型打印相应的参数，避免误导
    model_params_for_log = None
    if config.MODEL == 'LGB':
        model_params_for_log = config.LGBM_PARAMS
    elif config.MODEL == 'CAT':
        model_params_for_log = config.CAT_PARAMS
    elif config.MODEL == 'XGB':
        model_params_for_log = config.XGB_PARAMS
    else:
        model_params_for_log = {}
    logger.info(f"Using model: {config.MODEL}")
    logger.info(f"Model Parameters: {json.dumps(model_params_for_log, indent=4)}")

    # 1. 加载特征和标签
    feature_df, loaded_feature_name = features.load_features(feature_file_name, data_ids=data_ids)
    logger.info(f"Successfully loaded features from: {loaded_feature_name}")
    _, y_train = data.load_data(enhancement_ids=data_ids, return_dict=False)
    # 确保对齐
    common_index = feature_df.index.intersection(y_train.index)
    feature_df = feature_df.loc[common_index]
    y_train = y_train.loc[common_index]['structural_breakpoint'].astype(int)
    logger.info(f"训练数据已对齐. X shape: {feature_df.shape}, y shape: {y_train.shape}")
    
    # # 2. PCA降维
    # reducer = DimReducer(seg='left', reducer_params={'n_components': 42})
    # reducer.fit(feature_df)
    # reduced_feature_df, _ = reducer.extract(feature_df)
    # # 2. 最近邻
    # extractor = NeighborFeatureExtractor(metric='euclidean', stage='train')
    # extractor.fit(reduced_feature_df, y_train, n_neighbors=50)
    # extracted_feature_df, nn_features = extractor.extract(reduced_feature_df)
    # logger.info(f"最近邻特征提取完成，新增 {len(nn_features)} 个特征。")
    # for col in nn_features:
    #     null_ratio = extracted_feature_df[col].isnull().sum() / len(extracted_feature_df)
    #     zero_ratio = (extracted_feature_df[col] == 0).sum() / len(extracted_feature_df)
    #     logger.info(f"    - '{col}': 空值比例={null_ratio:.2%}, 零值比例={zero_ratio:.2%}")

    # 特征选择
    if len(config.REMAIN_FEATURES) > 0:
        feature_df = feature_df[config.REMAIN_FEATURES]
    # feature_df = pd.concat([feature_df, extracted_feature_df], axis=1)
    if feature_df is None:
        logger.error("特征加载失败，训练中止。")
        return None, None

    logger.info(f"--- 使用的特征列表 (共 {len(feature_df.columns)} 个) ---")
    logger.info(f"前50个特征: {feature_df.columns.tolist()[:50]}")
    logger.info("-" * min(50, len(str(feature_df.columns.tolist()))))
    
    # 3. 交叉验证
    logger.info("Starting 5-fold cross-validation with enhanced data strategy...")

    oof_preds = np.zeros(len(feature_df[0:10001]))
    models = []
    feature_importances = pd.DataFrame(index=feature_df.columns)
    permutation_results = pd.DataFrame(index=feature_df.columns)
    fold_metrics = []
    
    # 使用增强数据交叉验证策略
    cv_iterator = create_enhanced_cv_splits(feature_df, y_train, data_ids, config.CV_PARAMS)
    for fold, (train_idx, val_idx) in enumerate(cv_iterator):
        logger.info(f"--- Fold {fold+1}/{config.CV_PARAMS['n_splits']} ---")
        fold_start_time = time.time()

        X_train_fold, y_train_fold = feature_df.iloc[train_idx], y_train.iloc[train_idx]
        X_val_fold, y_val_fold = feature_df.iloc[val_idx], y_train.iloc[val_idx]
        logger.info(f"训练数据: {X_train_fold.shape}, 验证数据: {X_val_fold.shape}")

        # 配置模型
        if config.MODEL == 'LGB':
            model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
            callbacks = []
            if getattr(config, 'EARLY_STOPPING_ROUNDS', 0) and config.EARLY_STOPPING_ROUNDS > 0:
                callbacks.append(lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False))
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
                eval_names=['train', 'valid'],
                eval_metric='auc',
                callbacks=callbacks
            )
            train_auc = model.best_score_['train']['auc']
        elif config.MODEL == 'CAT':
            model = cat.CatBoostClassifier(**config.CAT_PARAMS)
            model.fit(
                X_train_fold, y_train_fold, 
                eval_set=[(X_val_fold, y_val_fold)],
                early_stopping_rounds=(config.EARLY_STOPPING_ROUNDS if getattr(config, 'EARLY_STOPPING_ROUNDS', 0) and config.EARLY_STOPPING_ROUNDS > 0 else None),
                verbose=False
            )
            train_preds = model.predict_proba(X_train_fold)[:, 1]
            # 确保y_train_fold是NumPy格式，兼容cuDF
            y_train_fold_numpy = y_train_fold.to_numpy() if hasattr(y_train_fold, 'to_numpy') else y_train_fold
            train_auc = roc_auc_score(y_train_fold_numpy, train_preds)
        elif config.MODEL == 'XGB':
            if getattr(config, 'EARLY_STOPPING_ROUNDS', 0) and config.EARLY_STOPPING_ROUNDS > 0:
                config.XGB_PARAMS['early_stopping_rounds'] = config.EARLY_STOPPING_ROUNDS
            model = xgb.XGBClassifier(**config.XGB_PARAMS)
            # 如果使用GPU且cudf可用，转换数据到GPU
            if CUDF_AVAILABLE and config.XGB_PARAMS.get('device') == 'cuda':
                logger.info(f"Fold {fold+1}: Using cuDF for GPU data processing")
                X_train_fold_gpu = cudf.DataFrame(X_train_fold)
                y_train_fold_gpu = cudf.Series(y_train_fold)
                X_val_fold_gpu = cudf.DataFrame(X_val_fold)
                y_val_fold_gpu = cudf.Series(y_val_fold)
                model.fit(
                    X_train_fold_gpu, y_train_fold_gpu, 
                    eval_set=[(X_train_fold_gpu, y_train_fold_gpu), (X_val_fold_gpu, y_val_fold_gpu)],
                    verbose=False
                )
            else:
                model.fit(
                    X_train_fold, y_train_fold, 
                    eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
                    verbose=False
                )
            train_preds = model.predict_proba(X_train_fold)[:, 1]
            train_auc = roc_auc_score(y_train_fold, train_preds)
        else:
            raise ValueError("Unknown config.MODEL")

        # 预测验证集
        preds = model.predict_proba(X_val_fold)[:, 1]
        if hasattr(preds, 'get'):
            preds = preds.get()
        
        oof_preds[val_idx] = preds
        models.append(model)
        feature_importances[f'fold_{fold+1}'] = model.feature_importances_
        
        fold_auc = roc_auc_score(y_val_fold, preds)
        logger.info(f"Fold {fold+1} Train AUC: {train_auc:.5f}, Val AUC: {fold_auc:.5f}")

        # 记录早停的 step（best_iteration）
        best_iteration = None
        if config.MODEL == 'LGB':
            best_iteration = getattr(model, 'best_iteration_', None)
        elif config.MODEL == 'CAT':
            try:
                best_iteration = model.get_best_iteration()
            except Exception:
                best_iteration = getattr(model, 'best_iteration_', None)
        elif config.MODEL == 'XGB':
            best_iteration = getattr(model, 'best_iteration', None)
        logger.info(f"Fold {fold+1} Early stopping step (best_iteration): {best_iteration}")

        # 保存到元数据结构中
        fold_metrics.append({
            'fold': fold + 1,
            'train_auc': float(train_auc),
            'val_auc': float(fold_auc),
            'best_iteration': int(best_iteration) if best_iteration is not None else None,
        })

        fold_duration = time.time() - fold_start_time
        logger.info(f"Fold {fold+1} finished in {fold_duration:.2f}s")

        # 可选地计算permutation importance
        if perm_imp:
            logger.info(f"Calculate Fold {fold+1} permutation importance...")
            perm_start_time = time.time()
            # 在验证集上计算permutation importance
            perm_result = permutation_importance(
                model, X_val_fold, y_val_fold,
                n_repeats=20,  # 可以根据需要调整重复次数
                random_state=config.SEED,
                scoring='roc_auc',
                n_jobs=-1
            )
            # 保存每个fold的permutation importance
            permutation_results[f'fold_{fold+1}'] = perm_result.importances_mean
            perm_duration = time.time() - perm_start_time
            logger.info(f"Fold {fold+1} permutation importance finished in {perm_duration:.2f}s")

    overall_oof_auc = roc_auc_score(y_train[0:10001], oof_preds)
    logger.info(f"Overall OOF AUC: {overall_oof_auc:.5f}")

    # 4. 创建带时间和AUC分数的输出文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    auc_str = f"{overall_oof_auc:.5f}".replace('.', '_')
    run_output_dir = config.OUTPUT_DIR / f'train_{timestamp}_auc_{auc_str}'
    run_output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"All outputs will be saved to: {run_output_dir}")

    # 5. 计算并保存特征重要性
    mean_importance = feature_importances.mean(axis=1)
    save_feature_importance(mean_importance, feature_df.columns, run_output_dir)
    logger.info("feature importance saved.")

    # 6. 保存模型
    if save_model:
        for i, model in tqdm(enumerate(models), total=len(models), desc="Saving models"):
            joblib.dump(model, run_output_dir / f'local_{config.MODEL}_model_fold_{i+1}.pkl')
        # extractor.save(run_output_dir / 'neighbor_extractor.pkl')
        logger.info("Models saved.")

    # 7. 可选地保存 OOF
    if save_oof:
        oof_df = pd.DataFrame({'id': feature_df.index, 'oof_preds': oof_preds})
        oof_df.to_csv(run_output_dir / 'oof_preds.csv', index=False)
        logger.info("OOF predictions saved.")
        
    # 8. 保存本次训练的元数据
    training_metadata = {
        "feature_file_used": loaded_feature_name,
        "features_used": feature_df.columns.tolist(),
        "model_name": config.MODEL,
        "model_params": model_params_for_log,
        "cv_params": config.CV_PARAMS,
        "oof_auc": overall_oof_auc,
        "fold_metrics": fold_metrics
    }
    with open(run_output_dir / 'training_metadata.json', 'w') as f:
        json.dump(training_metadata, f, indent=4)

    # 9. 保存permutation importance
    if perm_imp:
        save_permutation_importance(permutation_results, feature_df.columns, run_output_dir)
        logger.info("permutation importance saved.")

    duration = time.time() - start_time
    logger.info(f"训练流程结束，总耗时: {duration:.2f} 秒。")
    
    return models, overall_oof_auc 


def tune_hyperparameter(feature_file_name: str, data_ids: list = ["0"], n_trials: int = 50):
    """
    基于Optuna进行超参数调优
    """
    start_time = time.time()
    logger.info("Starting Optuna hyperparameter tuning...")

    # 1. 加载特征和标签
    feature_df, loaded_feature_name = features.load_features(feature_file_name, data_ids=data_ids)
    logger.info(f"Successfully loaded features from: {loaded_feature_name}")
    _, y_train = data.load_data(enhancement_ids=data_ids, return_dict=False)
    # 确保对齐
    common_index = feature_df.index.intersection(y_train.index)
    feature_df = feature_df.loc[common_index]
    y_train = y_train.loc[common_index]['structural_breakpoint'].astype(int)
    logger.info(f"训练数据已对齐. X shape: {feature_df.shape}, y shape: {y_train.shape}")

    # 特征选择
    if len(config.REMAIN_FEATURES) > 0:
        feature_df = feature_df[config.REMAIN_FEATURES]
    # feature_df = pd.concat([feature_df, extracted_feature_df], axis=1)
    if feature_df is None:
        logger.error("特征加载失败，训练中止。")
        return None, None

    logger.info(f"--- 使用的特征列表 (共 {len(feature_df.columns)} 个) ---")
    logger.info(feature_df.columns.tolist())
    logger.info("-" * min(50, len(str(feature_df.columns.tolist()))))
    
    def objective(trial):
        params = {
            # --- 基础设定 ---
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 2000, 8000),
            "learning_rate": trial.suggest_loguniform("learning_rate", 5e-4, 5e-2),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "random_state": config.SEED,
            "n_jobs": config.N_JOBS,

            # --- 正则化和采样 ---
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 50),
            "subsample": trial.suggest_uniform("subsample", 0.8, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 0, 10),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.8, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 3.0, 10.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 3.0, 10.0),
        }

        oof_preds = np.zeros(len(feature_df))
        skf = StratifiedKFold(**config.CV_PARAMS)
        cv_iterator = skf.split(feature_df, y_train)

        for fold, (train_idx, val_idx) in enumerate(cv_iterator):
            logger.info(f"Trial {trial.number} - Fold {fold+1}/{config.CV_PARAMS['n_splits']}")
            X_train_fold, y_train_fold = feature_df.iloc[train_idx], y_train.iloc[train_idx]
            X_val_fold, y_val_fold = feature_df.iloc[val_idx], y_train.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
                eval_names=['train', 'valid'],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )

            preds = model.predict_proba(X_val_fold)[:, 1]
            oof_preds[val_idx] = preds

            train_auc = model.best_score_['train']['auc']
            val_auc = roc_auc_score(y_val_fold, preds)
            logger.info(f"Fold {fold + 1} - Train AUC: {train_auc:.5f}, Val AUC: {val_auc:.5f}")

        overall_auc = roc_auc_score(y_train, oof_preds)
        logger.info(f"Trial {trial.number} - Overall OOF AUC: {overall_auc:.5f}")
        return overall_auc

    # 3. Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best Trial AUC: {study.best_value:.5f}")
    logger.info(f"Best Params: {study.best_params}")

    # 4. 创建带时间和AUC分数的输出文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    auc_str = f"{study.best_value:.5f}".replace('.', '_')
    run_output_dir = config.OUTPUT_DIR / f'train_{timestamp}_auc_{auc_str}'
    run_output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Optuna results saved to: {run_output_dir}")
    with open(run_output_dir / 'best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)

    duration = time.time() - start_time
    logger.info(f"Optuna tuning finished in {duration:.2f}s")
    
    return study.best_value