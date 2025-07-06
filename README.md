# 轻量级数据科学竞赛工作流

## 1. 核心理念

本工作流为快节奏的数据科学竞赛量身定制，核心是 **快速迭代** 和 **结果可追溯**。我们摒弃了复杂的版本控制（如Git分支），将所有精力集中在特征工程和模型训练上。

*   **特征集的演进**: 我们不维护单一的"主"特征文件，而是通过生成一系列带时间戳的特征文件来记录实验的每一步。最新的文件代表了当前最优的特征集。
*   **版本即文件**: 每个特征文件 (`features_YYYYMMDD_HHMMSS.parquet`) 都是一个独立的版本。想要回退或比较，只需在命令中指定不同的文件名即可。
*   **自动化日志**: 所有的操作（特征生成/删除、模型训练）都会生成带时间戳的、独立的日志文件。成功的训练日志会自动以CV分数重命名，方便快速定位。
*   **输出隔离**: 每次训练的产出（模型、OOF预测、日志、元数据）都保存在以时间戳和CV分数命名的独立文件夹中，确保实验结果清晰、不混淆。

---

## 2. 项目结构

```
ADIA-Lab-Structural-Break/
|
├── data/                       # 原始数据
|
├── experiment/                 # 核心实验代码
|   ├── backups/                # 特征文件的自动备份
|   ├── feature_dfs/            # 所有版本的时间戳特征文件 (e.g., features_20231029_153000.parquet)
|   ├── logs/                   # 所有运行日志
|   |   ├── feature_eng_...log  # 特征工程日志
|   |   └── train_...log        # 训练日志 (成功后会重命名加入CV分数)
|   ├── output/                 # 所有训练产出
|   |   └── 20231029_160000_cv_0.81234/ # 以"时间戳_cv_分数"命名的单次训练结果
|   |       ├── model_fold_0.lgb
|   |       ├── oof_preds.parquet
|   |       └── training_metadata.json
|   ├── config.py
|   ├── features.py             # **所有特征函数的定义之处**
|   ├── main.py                 # 命令行入口
|   ├── train.py                # 训练逻辑
|   └── utils.py                # 工具函数 (如日志)
|
├── notebooks/                  # EDA 和快速原型验证
└── README.md                   # 本文档
```

---

## 3. 日常工作流：一次完整的特征实验

假设你想添加一个新特征 `new_awesome_feature`。

#### 第1步: 在 `features.py` 中开发特征函数

在 `experiment/features.py` 文件中，添加你的特征函数，并用 `@register_feature` 装饰器标记。

```python
# experiment/features.py

# ... existing feature functions ...

@register_feature
def new_awesome_feature(df: pl.DataFrame) -> pl.DataFrame:
    """一个绝妙的新特征"""
    # ... 你的特征计算逻辑 ...
    # 必须返回一个包含 'series_id' 和新特征列的 DataFrame
    return df.select([
        'series_id',
        (pl.col('anglez').rolling_mean(50) * pl.col('enmo').rolling_mean(50)).alias('new_awesome_feature')
    ])
```

#### 第2步: 生成包含新特征的新版特征集

打开终端，运行 `gen-feats` 命令。这会基于最新的特征文件，加入你的新特征，并生成一个全新的、带时间戳的特征文件。

```bash
# 该命令会自动找到 experiment/feature_dfs/ 中最新的文件作为基础
# 然后添加 new_awesome_feature，并生成如 features_20231029_170000.parquet 的新文件
python -m experiment.main gen-feats --funcs new_awesome_feature
```
你会在 `experiment/feature_dfs/` 目录下看到新生成的文件。旧文件已被自动备份到 `experiment/backups/`。

#### 第3步: 使用新特征集进行训练

直接运行 `train` 命令。脚本会自动查找并使用 `feature_dfs` 目录中最新的特征文件进行训练。

```bash
# 自动使用最新的特征集进行训练
python -m experiment.main train --save-model --save-oof
```
*   `--save-model`: 保存训练好的模型文件。
*   `--save-oof`: 保存OOF（Out-of-Fold）预测结果。

#### 第4步: 评估结果

训练完成后，查看终端输出的CV分数。同时，你可以在文件系统中看到结果：

1.  **产出文件夹**: `experiment/output/` 下会出现一个新的文件夹，例如 `20231029_171500_cv_0.81500`。这里面包含了这次训练的所有产出。
2.  **成功日志**: `experiment/logs/` 下对应的训练日志会被重命名，例如 `train_20231029_171500_cv_0.81500.log`。

#### 第5步: 决策与同步

*   **实验成功 (CV分数提升)**:
    1.  **记录**: 在团队共享文档或 `CHANGELOG.md` 中记录你的贡献。
    2.  **同步**: `git add`, `git commit`, `git push` 你修改后的 `experiment/features.py` 文件。团队其他成员 `git pull` 后即可使用你的新特征。

*   **实验失败 (CV分数下降)**:
    1.  **无需手动回滚文件!** 你的失败尝试生成了一个新的特征文件，但之前的、效果更好的版本仍然安全地存在于 `feature_dfs` 目录中。
    2.  **代码清理**: 你可以从 `features.py` 中删除或注释掉失败的特征函数代码。
    3.  **基于旧版再实验**: 下次运行时，你可以告诉脚本基于某个指定的旧版本特征文件来继续工作。

---

## 4. 如何回滚 / 基于旧版本实验

如果最新的特征集效果不好，你想退回或基于某个历史版本进行新的实验，非常简单。

假设 `features_20231028_120000.parquet` 是一个已知的、效果很好的版本。

*   **只用它来训练**:
    ```bash
    python -m experiment.main train --feature-file experiment/feature_dfs/features_20231028_120000.parquet
    ```

*   **基于它来添加新特征**:
    ```bash
    python -m experiment.main gen-feats --funcs another_new_idea --base-feature-file experiment/feature_dfs/features_20231028_120000.parquet
    ```
    这会读取指定的旧文件，添加 `another_new_idea` 特征，然后生成一个全新的特征文件，而不会影响 `feature_dfs` 中的任何现有文件。

---

## 5. 版本追踪与复现

**任何一次训练的结果都是100%可复现的。**

如果你在 `output` 目录看到了一个高分结果，例如 `20231029_180000_cv_0.82000`，想知道它是怎么来的：

1.  找到对应的日志文件 `experiment/logs/train_20231029_180000_cv_0.82000.log`。
2.  打开该日志文件，里面清晰地记录了所有信息：
    *   **使用的特征文件**: 日志中会有一行 `INFO - Training with feature file: experiment/feature_dfs/features_xxxxxxxx_xxxxxx.parquet`。
    *   **模型参数**: 日志中记录了此次训练使用的所有模型超参数。
    *   **Git Commit Hash**: 记录了当时的代码版本。

有了这些信息，任何人都可以精确地复现这次高分提交。

---
## 6. 命令参考

*   **生成特征**: `python -m experiment.main gen-feats --funcs <func_name_1> <func_name_2> ...`
    *   `--base-feature-file <path>`: 指定一个基础特征文件，默认为最新。

*   **删除特征**: `python -m experiment.main del-feats --funcs <func_name_1> <func_name_2> ...`
    *   `--base-feature-file <path>`: 指定一个基础特征文件，默认为最新。

*   **模型训练**: `python -m experiment.main train`
    *   `--feature-file <path>`: 指定用于训练的特征文件，默认为最新。
    *   `--save-model`: Flag, 是否保存模型文件。
    *   `--save-oof`: Flag, 是否保存OOF预测文件。 