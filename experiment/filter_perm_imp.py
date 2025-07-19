import pandas as pd
import os
from . import config

def filter_and_save(version: str, path: str, top_k: list[int] = None, thresholds: list[float] = None):
    if top_k is None:
        top_k = [5, 10, 15]
    if thresholds is None:
        thresholds = [0.0005, 0.0004, 0.0003, 0.0002, 0.0001]

    imp_path = os.path.join(config.OUTPUT_DIR, version, path)
    output_file = os.path.join(config.OUTPUT_DIR, version, 'filtered_features_by_threshold.txt')

    df = pd.read_csv(imp_path, sep='\t')

    with open(output_file, 'w', encoding='utf-8') as f:
        for k in top_k:
            features = (
                df.sort_values('permutation_importance_mean', ascending=False)['feature']
                .head(k)
                .tolist()
            )
            f.write(f'# Top {k}\n')
            f.write('top_features = [\n')
            f.writelines([f"    '{feat}',\n" for feat in features])
            f.write(']\n\n')
        f.write('\n' + '*' * 50 + '\n')
        for th in thresholds:
            features = (
                df.loc[df['permutation_importance_mean'] > th]
                .sort_values('permutation_importance_mean', ascending=False)['feature']
                .tolist()
            )
            f.write(f'# Threshold: {th}\n')
            f.write(f'# Feature Num: {len(features)}\n')
            f.write('filtered_features = [\n')
            f.writelines([f"    '{feat}',\n" for feat in features])
            f.write(']\n\n')

    print(f"[filter_perm_imp] 特征筛选结果已保存至 {output_file}")
