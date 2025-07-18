import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm.auto import tqdm

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
    
    
def train_and_evaluate(feature_file_name: str, save_oof: bool = False, save_model: bool = False, perm_imp: bool = False):
    """
    加载指定的特征文件和标签，进行交叉验证，并根据参数选择性保存产出物。
    """
    start_time = time.time()
    logger.info("Starting training and evaluation pipeline...")
    logger.info(f"Model Parameters: {json.dumps(config.LGBM_PARAMS, indent=4)}")

    # 1. 加载特征和标签
    feature_df, loaded_feature_name = features.load_features(feature_file_name)
    logger.info(f"Successfully loaded features from: {loaded_feature_name}")
    _, y_train = data.load_data()
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
    feature_df = feature_df[config.REMAIN_FEATURES]
    # feature_df = pd.concat([feature_df, extracted_feature_df], axis=1)
    if feature_df is None:
        logger.error("特征加载失败，训练中止。")
        return None, None

    logger.info(f"--- 使用的特征列表 (共 {len(feature_df.columns)} 个) ---")
    logger.info(feature_df.columns.tolist())
    logger.info("-" * min(50, len(str(feature_df.columns.tolist()))))
    
    # 3. 交叉验证
    logger.info("Starting 5-fold cross-validation with LightGBM...")

    skf = StratifiedKFold(**config.CV_PARAMS)

    oof_preds = np.zeros(len(feature_df))
    models = []
    feature_importances = pd.DataFrame(index=feature_df.columns)
    permutation_results = pd.DataFrame(index=feature_df.columns)
    
    cv_iterator = skf.split(feature_df, y_train)
    for fold, (train_idx, val_idx) in enumerate(cv_iterator):
        logger.info(f"--- Fold {fold+1}/{config.CV_PARAMS['n_splits']} ---")
        fold_start_time = time.time()

        X_train_fold, y_train_fold = feature_df.iloc[train_idx], y_train.iloc[train_idx]
        X_val_fold, y_val_fold = feature_df.iloc[val_idx], y_train.iloc[val_idx]

        model = lgb.LGBMClassifier(**config.LGBM_PARAMS)

        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
            eval_names=['train', 'valid'],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        preds = model.predict_proba(X_val_fold)[:, 1]
        oof_preds[val_idx] = preds
        models.append(model)
        feature_importances[f'fold_{fold+1}'] = model.feature_importances_
        
        train_auc = model.best_score_['train']['auc']
        fold_auc = roc_auc_score(y_val_fold, preds)
        logger.info(f"Fold {fold+1} Train AUC: {train_auc:.5f}, Val AUC: {fold_auc:.5f}")

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
                random_state=42,
                scoring='roc_auc',
                n_jobs=-1
            )
            # 保存每个fold的permutation importance
            permutation_results[f'fold_{fold+1}'] = perm_result.importances_mean
            perm_duration = time.time() - perm_start_time
            logger.info(f"Fold {fold+1} permutation importance finished in {perm_duration:.2f}s")

    overall_oof_auc = roc_auc_score(y_train, oof_preds)
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
            joblib.dump(model, run_output_dir / f'model_fold_{i+1}.pkl')
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
        "model_params": config.LGBM_PARAMS,
        "cv_params": config.CV_PARAMS,
        "oof_auc": overall_oof_auc
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