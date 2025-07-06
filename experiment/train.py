import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import logging
import os
import matplotlib.pyplot as plt
from pathlib import Path

from . import config, features, data

logger = logging.getLogger(__name__)

def train_and_evaluate():
    """
    加载持久化的特征和标签，然后进行5折交叉验证训练和评估。
    """
    logger.info("Starting training and evaluation pipeline...")

    # 1. 加载特征和标签
    try:
        feature_df = features.load_features()
        _, y_train = data.load_data()
    except FileNotFoundError as e:
        logger.error(f"无法开始训练: {e}")
        return

    X_train_features = feature_df
    y_train_target = y_train.loc[X_train_features.index]['structural_breakpoint'].astype(int)

    # 确保对齐
    common_index = X_train_features.index.intersection(y_train_target.index)
    X_train_features = X_train_features.loc[common_index]
    y_train_target = y_train_target.loc[common_index]
    
    logger.info(f"训练数据已对齐. X shape: {X_train_features.shape}, y shape: {y_train_target.shape}")
    logger.info("Starting 5-fold cross-validation with LightGBM...")

    skf = StratifiedKFold(**config.CV_PARAMS)

    oof_preds = np.zeros(len(X_train_features))
    models = []
    # 使用 all_feature_importances 替换 feature_importances 以避免命名冲突
    all_feature_importances = pd.DataFrame(index=X_train_features.columns)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_features, y_train_target)):
        logger.info(f"--- Fold {fold+1}/{config.CV_PARAMS['n_splits']} ---")

        X_train_fold, y_train_fold = X_train_features.iloc[train_idx], y_train_target.iloc[train_idx]
        X_val_fold, y_val_fold = X_train_features.iloc[val_idx], y_train_target.iloc[val_idx]

        model = lgb.LGBMClassifier(**config.LGBM_PARAMS)

        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(100, verbose=False)] # 静默回调
        )

        preds = model.predict_proba(X_val_fold)[:, 1]
        oof_preds[val_idx] = preds
        models.append(model)
        fold_auc = roc_auc_score(y_val_fold, preds)
        logger.info(f"Fold {fold+1} AUC: {fold_auc:.5f}")

        fold_importance = pd.DataFrame({
            'importance': model.feature_importances_
        }, index=X_train_features.columns)
        all_feature_importances[f'fold_{fold+1}'] = fold_importance['importance']

    overall_oof_auc = roc_auc_score(y_train_target, oof_preds)
    logger.info(f"Overall OOF AUC: {overall_oof_auc:.5f}")

    # 保存 OOF 预测
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    oof_df = pd.DataFrame({'id': X_train_features.index, 'oof_preds': oof_preds})
    oof_path = config.OUTPUT_DIR / 'oof_preds.csv'
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"OOF predictions saved to {oof_path}")

    # 保存特征重要性图
    plot_feature_importance(all_feature_importances, config.OUTPUT_DIR)
    
    return models, overall_oof_auc

def plot_feature_importance(importances_df, output_dir):
    """绘制并保存特征重要性图表"""
    mean_importance = importances_df.mean(axis=1).sort_values(ascending=False)
    
    plt.figure(figsize=(12, max(6, len(mean_importance) // 4)))
    mean_importance.head(50).sort_values().plot(kind='barh') # 只显示 top 50
    plt.title('Average Feature Importance across Folds (Top 50)')
    plt.xlabel('Mean Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    
    fig_path = output_dir / 'feature_importance.png'
    plt.savefig(fig_path)
    logger.info(f"Feature importance plot saved to {fig_path}")
    plt.close() 