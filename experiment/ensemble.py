import json
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

import lightgbm as lgb
import catboost as cat
import xgboost as xgb

from sklearn.metrics import roc_auc_score

from . import config, features, data
from .train import create_enhanced_cv_splits

# logger 将由 main.py 在运行时注入
logger = None


def _build_model(model_type: str, params: dict):
    if model_type == 'LGB':
        return lgb.LGBMClassifier(**params)
    if model_type == 'CAT':
        return cat.CatBoostClassifier(**params)
    if model_type == 'XGB':
        return xgb.XGBClassifier(**params)
    raise ValueError(f"Unknown model type: {model_type}")


def _predict_proba(model, X):
    preds = model.predict_proba(X)[:, 1]
    # catboost/cupy 返回的可能是 GPU 对象
    return preds.get() if hasattr(preds, 'get') else preds


def _load_fold_model(pretrained_dir, model_type: str, fold_idx: int):
    import joblib
    path = pretrained_dir / f'local_{model_type}_model_fold_{fold_idx + 1}.pkl'
    if not path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {path}")
    return joblib.load(path)


def _get_params_from_config(params_name: str, overrides: dict | None = None) -> dict:
    base_params = deepcopy(getattr(config, params_name)) if params_name else {}
    if overrides:
        base_params.update(overrides)
    return base_params


def _aggregate(pred_list: list[np.ndarray], weights: list[float], method: str) -> np.ndarray:
    if method == 'weighted_mean':
        weights_arr = np.array(weights, dtype=float)
        weights_arr = weights_arr / weights_arr.sum()
        stacked = np.stack(pred_list, axis=1)
        return (stacked * weights_arr).sum(axis=1)
    raise ValueError(f"Unknown aggregation method: {method}")


def run_ensemble(
    feature_file_name: str | None = None,
    data_ids: list[str] = None,
    use_pretrained: bool | None = None,
    save_oof: bool = True,
    save_model: bool = True,
):
    """
    运行可扩展的模型集成。

    - 当 use_pretrained=True 时：不训练，仅按 5 折进行推理融合。
    - 当 use_pretrained=False 时：对每个基模型执行 5 折训练与 OOF 预测，并保存模型与 OOF；随后进行融合并评估。
    - 支持同模型不同参数（通过配置中的 params 覆盖）。
    """
    start_time = time.time()

    if data_ids is None:
        data_ids = ["0"]
    if use_pretrained is None:
        use_pretrained = config.ENSEMBLE.get('use_pretrained', False)

    logger.info("Starting ensemble pipeline...")
    logger.info(f"Using pretrained: {use_pretrained}")
    logger.info(f"Ensemble plan: {json.dumps(config.ENSEMBLE, indent=2, default=str)}")

    # 1) 加载特征与标签，并对齐
    feature_df, loaded_feature_name = features.load_features(feature_file_name, data_ids=data_ids)
    logger.info(f"Successfully loaded features from: {loaded_feature_name}")
    _, y_train_df = data.load_data(enhancement_ids=data_ids, return_dict=False)

    common_index = feature_df.index.intersection(y_train_df.index)
    feature_df = feature_df.loc[common_index]
    y_train = y_train_df.loc[common_index]['structural_breakpoint'].astype(int)
    logger.info(f"训练数据已对齐. X shape: {feature_df.shape}, y shape: {y_train.shape}")

    # 仅在 REMAIN_FEATURES 非空时筛选
    if len(config.REMAIN_FEATURES) > 0:
        feature_df = feature_df[config.REMAIN_FEATURES]

    # 2) 交叉验证迭代器（增强数据策略），但评估/OOF 仅针对原始 0..10000
    cv_iterator = create_enhanced_cv_splits(feature_df, y_train, data_ids, config.CV_PARAMS)

    # 存储每个基模型的 OOF 预测，仅针对原始部分（与 train.py 保持一致）
    original_len = len(feature_df[0:10001])
    base_model_oof = {}
    base_model_metrics = {}
    fold_indices_cache = []  # 缓存每折的 (train_idx, val_idx)

    # 预先跑一遍 iterator 收集索引（多模型共用同一折划分）
    for train_idx, val_idx in cv_iterator:
        fold_indices_cache.append((train_idx, val_idx))

    # 3) 针对每个基模型执行训练/加载与 OOF 推断
    from pathlib import Path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_dir = config.OUTPUT_DIR / f'ensemble_{timestamp}'
    run_output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"All ensemble outputs will be saved to: {run_output_dir}")

    for m_cfg in config.ENSEMBLE['models']:
        name = m_cfg['name']
        mtype = m_cfg['type']
        params_name = m_cfg.get('params_name')
        params_overrides = m_cfg.get('params', {})
        pretrained_dir = Path(m_cfg['pretrained_dir']) if m_cfg.get('pretrained_dir') else None

        logger.info(f"=== Base model: {name} ({mtype}) ===")
        params = _get_params_from_config(params_name, params_overrides)
        logger.info(f"Params: {json.dumps(params, indent=2)}")

        # 每个基模型独立的保存目录
        model_out_dir = run_output_dir / f'model_{name}'
        model_out_dir.mkdir(exist_ok=True, parents=True)

        oof_preds = np.zeros(original_len)
        fold_metrics = []
        models = []

        # 遍历折
        for fold, (train_idx, val_idx) in enumerate(fold_indices_cache):
            X_train_fold, y_train_fold = feature_df.iloc[train_idx], y_train.iloc[train_idx]
            X_val_fold, y_val_fold = feature_df.iloc[val_idx], y_train.iloc[val_idx]

            if use_pretrained:
                if not pretrained_dir:
                    raise ValueError(f"Model {name} set to use_pretrained but pretrained_dir is None")
                model = _load_fold_model(pretrained_dir, mtype, fold)
                logger.info(f"[Fold {fold+1}] Loaded pretrained model from {pretrained_dir}")
            else:
                model = _build_model(mtype, params)

                # 早停设置（与 train.py 保持一致的行为）
                if mtype == 'LGB':
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
                elif mtype == 'CAT':
                    model.fit(
                        X_train_fold, y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        early_stopping_rounds=(config.EARLY_STOPPING_ROUNDS if getattr(config, 'EARLY_STOPPING_ROUNDS', 0) and config.EARLY_STOPPING_ROUNDS > 0 else None),
                        verbose=False
                    )
                elif mtype == 'XGB':
                    fit_params = {
                        'eval_set': [(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
                        'verbose': False,
                    }
                    if getattr(config, 'EARLY_STOPPING_ROUNDS', 0) and config.EARLY_STOPPING_ROUNDS > 0:
                        fit_params['early_stopping_rounds'] = config.EARLY_STOPPING_ROUNDS
                    model.fit(X_train_fold, y_train_fold, **fit_params)
                else:
                    raise ValueError(f"Unknown model type: {mtype}")

            # 训练集 AUC（仅训练模式可获取稳定的 train AUC）
            if not use_pretrained:
                if mtype == 'LGB':
                    train_auc = model.best_score_['train']['auc']
                else:
                    train_preds = _predict_proba(model, X_train_fold)
                    train_auc = roc_auc_score(y_train_fold, train_preds)
            else:
                train_auc = None

            # 验证集预测
            val_preds = _predict_proba(model, X_val_fold)
            val_auc = roc_auc_score(y_val_fold, val_preds)

            # 仅写入原始段的 OOF（val_idx 本身已仅包含原始数据，根据 create_enhanced_cv_splits 的设计）
            oof_preds[val_idx] = val_preds

            # 记录模型
            models.append(model)

            # 记录早停步数
            if mtype == 'LGB':
                best_iteration = getattr(model, 'best_iteration_', None)
            elif mtype == 'CAT':
                try:
                    best_iteration = model.get_best_iteration()
                except Exception:
                    best_iteration = getattr(model, 'best_iteration_', None)
            elif mtype == 'XGB':
                best_iteration = getattr(model, 'best_iteration', None)
            else:
                best_iteration = None

            logger.info(f"[{name}] Fold {fold+1} Train AUC: {train_auc if train_auc is not None else float('nan'):.5f}, Val AUC: {val_auc:.5f}")
            fold_metrics.append({
                'fold': fold + 1,
                'train_auc': float(train_auc) if train_auc is not None else None,
                'val_auc': float(val_auc),
                'best_iteration': int(best_iteration) if best_iteration is not None else None,
            })

            # 保存模型
            if (not use_pretrained) and save_model:
                import joblib
                joblib.dump(model, model_out_dir / f'local_{mtype}_model_fold_{fold+1}.pkl')

        # 单模型 OOF AUC
        model_oof_auc = roc_auc_score(y_train[0:original_len], oof_preds)
        logger.info(f"[{name}] Overall OOF AUC: {model_oof_auc:.5f}")
        base_model_oof[name] = oof_preds
        base_model_metrics[name] = {
            'oof_auc': float(model_oof_auc),
            'fold_metrics': fold_metrics,
            'type': mtype,
            'params': params,
        }

        # 保存单模型 OOF
        if save_oof:
            pd.DataFrame({'id': feature_df.index[:original_len], f'oof_{name}': oof_preds}).to_csv(
                model_out_dir / f'oof_{name}.csv', index=False
            )

        # 保存单模型元数据
        with open(model_out_dir / 'model_metadata.json', 'w') as f:
            json.dump(base_model_metrics[name], f, indent=4)

    # 4) 融合
    weights = [m['weight'] for m in config.ENSEMBLE['models']]
    names_in_order = [m['name'] for m in config.ENSEMBLE['models']]
    pred_list = [base_model_oof[name] for name in names_in_order]
    agg_method = config.ENSEMBLE.get('aggregation', 'weighted_mean')
    ensemble_oof = _aggregate(pred_list, weights, agg_method)
    ensemble_auc = roc_auc_score(y_train[0:original_len], ensemble_oof)
    logger.info(f"[Ensemble] OOF AUC: {ensemble_auc:.5f}")

    # 5) 保存整体输出
    oof_out = pd.DataFrame({'id': feature_df.index[:original_len], 'ensemble_oof': ensemble_oof})
    for name in names_in_order:
        oof_out[f'oof_{name}'] = base_model_oof[name]
    if save_oof:
        oof_out.to_csv(run_output_dir / 'oof_ensemble_and_bases.csv', index=False)

    # 保存集成元数据
    ensemble_metadata = {
        'feature_file_used': loaded_feature_name,
        'features_used': feature_df.columns.tolist(),
        'cv_params': config.CV_PARAMS,
        'base_models': base_model_metrics,
        'aggregation': agg_method,
        'weights': weights,
        'ensemble_oof_auc': float(ensemble_auc),
        'use_pretrained': use_pretrained,
        'data_ids': data_ids,
    }
    with open(run_output_dir / 'ensemble_metadata.json', 'w') as f:
        json.dump(ensemble_metadata, f, indent=4)

    # 将最终 AUC 体现在目录名中
    auc_str = f"{ensemble_auc:.5f}".replace('.', '_')
    final_dir = run_output_dir.with_name(f"{run_output_dir.name}_auc_{auc_str}")
    run_output_dir.rename(final_dir)
    logger.info(f"All ensemble outputs saved to: {final_dir}")

    duration = time.time() - start_time
    logger.info(f"集成流程结束，总耗时: {duration:.2f} 秒。")

    return {
        'run_output_dir': final_dir,
        'ensemble_auc': ensemble_auc,
        'base_model_metrics': base_model_metrics,
    }


