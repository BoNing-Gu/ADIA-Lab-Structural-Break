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
|   |   └── train_20231029_160000_cv_0.81234/ # 以"时间戳_auc_分数"命名的单次训练结果
|   |       ├── model_fold_1.txt
|   |       ├── oof_preds.csv
|   |       ├── training_metadata.json
|   |       └── feature_importance.png # 特征重要性图
|   ├── config.py
|   ├── features.py             # **所有激活特征函数的定义之处**
|   ├── features_deprecated.py  # **已废弃或实验失败的特征函数**
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
def new_awesome_feature(u: pd.DataFrame) -> dict:
    """一个绝妙的新特征，返回一个字典"""
    # ... 你的特征计算逻辑 ...
    # 必须返回一个包含新特征的字典
    return {'new_awesome_feature': 42}
```

#### 第1.5步: (可选) 将其设为实验性特征
如果你不确定这个特征是否有效，可以先将其加入 `experiment/config.py` 的 `EXPERIMENTAL_FEATURES` 列表。

```python
# experiment/config.py
EXPERIMENTAL_FEATURES = [
    'new_awesome_feature',
]
```
这样，默认的 `gen-feats` 命令会跳过它。只有当你通过 `--funcs` 参数明确指定它时，它才会被生成。这有助于保持主特征集的稳定。

#### 第2步: 生成包含新特征的新版特征集
打开终端，运行 `gen-feats` 命令并用 `--funcs` 指定要运行的函数。

```bash
# 基于最新的特征文件，加入 new_awesome_feature
# 这会生成一个全新的、带时间戳的特征文件
python -m experiment.main gen-feats --funcs new_awesome_feature
```
新生成的日志会包含新特征的**生成耗时、空值率、零值率**等详细信息，方便快速诊断。

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

1.  **产出文件夹**: `experiment/output/` 下会出现一个新的文件夹，例如 `train_20250707_140000_auc_0.69000`。
2.  **特征重要性图**: 在该文件夹内，你会找到一张 `feature_importance.png` 图，它可视化了所有特征的平均重要性，**高度会自适应调整**，确保所有特征名清晰可见。
3.  **成功日志**: `experiment/logs/` 下对应的训练日志会被重命名，例如 `train_20250707_140000_auc_0.69000.log`。日志中包含了**训练总耗时**和**使用的全部特征列表**。

#### 第5步: 决策与同步

*   **实验成功 (CV分数提升)**:
    1.  **确认**: 如果之前是实验性特征，记得从 `config.py` 的 `EXPERIMENTAL_FEATURES` 列表中移除它，使其成为核心特征。
    2.  **同步**: `git add`, `git commit`, `git push` 你修改过的 `experiment/` 目录下的相关文件。

*   **实验失败 (CV分数下降)**:
    1.  **归档代码**: 将 `features.py` 中失败的特征函数代码，完整地剪切并粘贴到 `experiment/features_deprecated.py` 文件中，并**移除函数头顶的 `@register_feature` 装饰器**。
    2.  **清理特征文件**: 使用 `del-feats` 命令从最新的特征文件中移除这些失败的特征列。
        ```bash
        # 假设最新的文件是 ...12345.parquet，要删除的特征由 new_awesome_feature 函数生成
        python -m experiment.main del-feats --base-file features_...12345.parquet --funcs new_awesome_feature
        ```
    这会生成一个不含失败特征的、更干净的新版本特征文件，你的工作区也随之回滚。

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

*   **生成特征**: `python -m experiment.main gen-feats [--funcs <func_name_1> ...]`
    *   `--funcs`: 指定要生成的一个或多个特征函数。**如果省略，则生成所有非实验性特征**。
    *   `--base-file <filename>`: 指定一个基础特征文件名进行更新，默认为最新。

*   **删除特征**: `python -m experiment.main del-feats --funcs <func_name_1> ... --base-file <filename>`
    *   `--funcs`: 指定要删除的特征**函数名**。脚本会自动找到该函数生成的所有列并删除它们。
    *   `--base-file <filename>`: **必须**指定要操作的基础特征文件名。

*   **模型训练**: `python -m experiment.main train`
    *   `--feature-file <path>`: 指定用于训练的特征文件，默认为最新。
    *   `--save-model`: Flag, 是否保存模型文件。
    *   `--save-oof`: Flag, 是否保存OOF预测文件。 