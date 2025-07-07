import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import logging
import json
import time
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from . import config, features, data

# logger 将由 main.py 在运行时注入
logger = None

def plot_feature_importance(feature_importances, feature_names, output_dir):
    """绘制并保存特征重要性图"""
    df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
    df = df.sort_values('importance', ascending=False)
    
    # 动态调整图形高度
    num_features = len(df)
    fig_height = max(10, num_features * 0.25)
    
    plt.figure(figsize=(12, fig_height))
    sns.barplot(x='importance', y='feature', data=df)
    plt.title('Feature Importance (averaged over folds)')
    plt.tight_layout() # 自动调整布局，防止标签重叠
    
    save_path = output_dir / 'feature_importance.png'
    plt.savefig(save_path)
    plt.close()
    logger.info(f"特征重要性图已保存到: {save_path}")

def train_and_evaluate(feature_file_name: str, save_oof: bool = False, save_model: bool = False):
    """
    加载指定的特征文件和标签，进行交叉验证，并根据参数选择性保存产出物。
    """
    start_time = time.time()
    logger.info("Starting training and evaluation pipeline...")
    logger.info(f"Model Parameters: {json.dumps(config.LGBM_PARAMS, indent=4)}")

    # 1. 加载特征和标签
    feature_df, loaded_feature_name = features.load_features(feature_file_name)
    if feature_df is None:
        logger.error("特征加载失败，训练中止。")
        return None, None

    logger.info(f"Successfully loaded features from: {loaded_feature_name}")
    
    _, y_train = data.load_data()

    # 确保对齐
    common_index = feature_df.index.intersection(y_train.index)
    feature_df = feature_df.loc[common_index]
    y_train = y_train.loc[common_index]['structural_breakpoint'].astype(int)
    
    logger.info(f"训练数据已对齐. X shape: {feature_df.shape}, y shape: {y_train.shape}")
    logger.info(f"--- 使用的特征列表 (共 {len(feature_df.columns)} 个) ---")
    logger.info(feature_df.columns.tolist())
    logger.info("-" * (len(str(feature_df.columns.tolist()))))
    
    logger.info("Starting 5-fold cross-validation with LightGBM...")

    skf = StratifiedKFold(**config.CV_PARAMS)

    oof_preds = np.zeros(len(feature_df))
    models = []
    feature_importances = pd.DataFrame(index=feature_df.columns)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(feature_df, y_train)):
        logger.info(f"--- Fold {fold+1}/{config.CV_PARAMS['n_splits']} ---")

        X_train_fold, y_train_fold = feature_df.iloc[train_idx], y_train.iloc[train_idx]
        X_val_fold, y_val_fold = feature_df.iloc[val_idx], y_train.iloc[val_idx]

        model = lgb.LGBMClassifier(**config.LGBM_PARAMS)

        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        preds = model.predict_proba(X_val_fold)[:, 1]
        oof_preds[val_idx] = preds
        models.append(model)
        feature_importances[f'fold_{fold+1}'] = model.feature_importances_
        fold_auc = roc_auc_score(y_val_fold, preds)
        logger.info(f"Fold {fold+1} AUC: {fold_auc:.5f}")

    overall_oof_auc = roc_auc_score(y_train, oof_preds)
    logger.info(f"Overall OOF AUC: {overall_oof_auc:.5f}")

    # 2. 创建带时间和AUC分数的输出文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    auc_str = f"{overall_oof_auc:.5f}".replace('.', '_')
    run_output_dir = config.OUTPUT_DIR / f'train_{timestamp}_auc_{auc_str}'
    run_output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"所有产出物将保存到: {run_output_dir}")

    # 3. 计算并绘制特征重要性
    mean_importance = feature_importances.mean(axis=1)
    plot_feature_importance(mean_importance, feature_df.columns, run_output_dir)

    # 4. 保存模型
    if save_model:
        for i, model in enumerate(models):
            joblib.dump(model, run_output_dir / f'model_fold_{i+1}.pkl')
        logger.info("Models saved.")

    # 5. 可选地保存 OOF
    if save_oof:
        oof_df = pd.DataFrame({'id': feature_df.index, 'oof_preds': oof_preds})
        oof_df.to_csv(run_output_dir / 'oof_preds.csv', index=False)
        logger.info("OOF predictions saved.")
        
    # 6. 保存本次训练的元数据
    training_metadata = {
        "feature_file_used": loaded_feature_name,
        "features_used": feature_df.columns.tolist(),
        "model_params": config.LGBM_PARAMS,
        "cv_params": config.CV_PARAMS,
        "oof_auc": overall_oof_auc
    }
    with open(run_output_dir / 'training_metadata.json', 'w') as f:
        json.dump(training_metadata, f, indent=4)

    duration = time.time() - start_time
    logger.info(f"训练流程结束，总耗时: {duration:.2f} 秒。")
    
    return models, overall_oof_auc 