# 轻量级数据科学竞赛工作流 （For ADIA Lab Structural Break）

中文版本 | [English](README.md)

## 特征工程方案

本Repo包含了我们为ADIA Lab Structural Break Competition构建的实验管道。查看我们的Notion文档以获取详细的特征工程解决方案。  
[特征工程方案](https://messy-plain-a65.notion.site/ADIA25-211402a1a1b780428471eaed714e285b?source=copy_link)

![ADIA LAB STRUCTRAL BREAK](figs/ADIA_LAB_STRUCTRAL_BREAK.png)

> `experiment/`是ML方法的实验管道、`submit_onlinetrain.ipynb`是用于提交在线训练模型的notebook（LB AUC 88.60%）。此外，`UTime.py`、`PatchCrossEncoder.py`是使用DL方法的尝试（性能较差，未被采纳）。

## 1. 核心理念

本工作流为快节奏的数据科学竞赛量身定制，核心是 **快速迭代** 和 **结果可追溯**。与git配合使用，确保每次实验的特征集和模型都能被版本化、回溯和共享。

*   **特征集的演进**: 我们不维护单一的"主"特征文件，而是通过生成一系列带时间戳的特征文件来记录实验的每一步。最新的文件代表了当前最优的特征集。
*   **版本即文件**: 每个特征文件 (`features_YYYYMMDD_HHMMSS.parquet`) 都是一个独立的版本。想要回退或比较，只需在命令中指定不同的文件名即可。
*   **自动化日志**: 所有的操作（特征生成/删除、模型训练）都会生成带时间戳的、独立的日志文件。成功的训练日志会自动以CV分数重命名，方便快速定位。
*   **输出隔离**: 每次训练的产出（模型、OOF预测、日志、元数据）都保存在以时间戳和CV分数命名的独立文件夹中，确保实验结果清晰、不混淆。

---

## 2. 项目结构

```
ADIA-Lab-Structural-Break/
├── experiment/                   # 主要实验模块
│   ├── config.py                 # 配置文件
│   ├── data.py                   # 数据加载和预处理
│   ├── features.py               # **所有激活特征函数的定义之处**
│   ├── features_deprecated.py    # **已废弃或实验失败的特征函数**
│   ├── filter.py                 # 特征过滤
│   ├── interactions.py           # 特征交互
│   ├── model.py                  # 实验性模型定义
│   ├── train.py                  # 训练和评估（含增强数据交叉验证）
│   ├── main.py                   # 主程序入口
│   ├── utils.py                  # 工具函数
│   ├── logs/                     # 日志文件
│   │   ├── feature_logs/         # 特征工程日志
│   │   └── training_logs/        # 训练日志
│   └── output/                   # 输出结果（按时间戳和AUC）
├── submit_onlinetrain.ipynb      # 提交文件
├── requirements.txt              # 依赖包
├── .gitignore                    # Git忽略文件
└── README.md                     # 项目说明
```

---

## 3. 工作流说明

#### 第1步：数据增强（可选，我们没有找到可以提升模型性能的增强方法）

在 `experiment/data.py` 文件中，添加数据增强函数，并用 `@register_data_enhancement` 装饰器标记。

```bash
# 生成所有注册好的增强数据
python -m experiment.main data-aug
```

然后在 `experiment/config.py` 中，添加数据增强函数的装饰器id到 `ENHANCEMENT_IDS` 列表中，这将标记哪些增强数据会在 `gen-feats` 命令中生成特征。

> 通过数据增强，我们可以增加训练样本数量（=原始数据集条数*增强个数）。

#### 第2步: 信号变换与特征工程

在 `experiment/features.py` 文件中，添加信号转换函数，并用 `@register_transform` 装饰器标记，用于对原始信号进行额外的变换。  
在 `experiment/features.py` 文件中，添加特征函数，并用 `@register_feature` 装饰器标记，用于从原始和变换后的信号中提取特征。

```bash
# 为所有原始和增强数据进行所有注册好的变换，然后生成所有注册好的特征
python -m experiment.main gen-feats
```

> 通过信号变换与特征工程，我们可以增加训练样本特征维度（=变换数*特征数）。

> 更多说明
> 1. 生成的特征文件会保存在 `feature_dfs/` 目录下，文件名格式为 `features_YYYYMMDD_HHMMSS.parquet`、`features_YYYYMMDD_HHMMSS_id_{x}.parquet`。无id后缀的文件记录特征的元信息，有id后缀的文件为对应各个增强数据的特征，`id_0`表示原始数据的特征。
> 2. 运行 `gen-feats` 命令并用 `--trans`和`--funcs` 参数指定要运行的变换函数和特征函数。这有助于基于feature_dfs/中最新的特征文件添加更多特征。
> 3. 如果你不确定这个特征是否有效，可以先将其加入 `experiment/config.py` 的 `EXPERIMENTAL_FEATURES` 列表。这样，默认的 `gen-feats` 命令会跳过它。只有当你通过 `--funcs` 参数明确指定它时，它才会被生成。这有助于保持主特征集的稳定。

#### 第3步：筛选

使用`experiment/config.py` 的 `REMAIN_FEATURES` 列表管理哪些特征是进一步训练所应使用的特征（筛选操作会输出恰当的结果文件，包含各个阈值的特征列表）。

生成特征后，执行相关性剔除，筛选结果呈现在 `./experiment/output/filter_{xxx}` 中，`{xxx}`是特征文件名（不含后缀）。  

```bash
# 执行相关性筛选
python -m experiment.main filter corr
```

训练后，使用feature importance / permutation importance筛选特征，请指定训练版本，筛选结果会保存在 `./experiment/output/filter_{xxx}` 中。

```bash
# 执行特征重要性筛选
python -m experiment.main filter feature-imp --train-version xxx
# 执行permutation importance筛选
python -m experiment.main filter perm-imp --train-version xxx
```

#### 第4步: 创建交互项

通过 `experiment/config.py` 的 `TOP_FEATURES` 列表中指定要交互的特征，`gen-interactions` 命令会自动生成这些特征的交互项，通过 `--sqmul --add --sub --div` 等参数指定要生成的交互项类型。

```bash
python -m experiment.main gen-interactions --sqmul --add --sub --div
```

#### 第5步: 训练

直接运行 `train` 命令。脚本会自动查找并使用 `feature_dfs` 目录中最新的特征文件进行训练。

```bash
# 自动使用最新的特征集进行训练
python -m experiment.main train --train-data-ids 0 --perm-imp --save-model --save-oof
```

*   `--train-data-ids 0 1 2`: 指定用于训练的数据id，可以接受多个值，默认值为id`0`的原始数据集。
*   `--perm-imp`: 计算permutation importance。
*   `--save-model`: 保存训练好的模型文件。
*   `--save-oof`: 保存OOF（Out-of-Fold）预测结果。

#### 第6步: 评估结果

训练完成后，查看终端输出的CV分数。同时，你可以在文件系统中看到结果：

1.  **产出文件夹**: `experiment/output/` 下会出现一个新的文件夹，例如 `train_20250707_140000_auc_0.69000`。
2.  **成功日志**: `experiment/logs/` 下对应的训练日志会被重命名，例如 `train_20250707_140000_auc_0.69000.log`。日志中包含了**训练总耗时**和**使用的全部特征列表**。

#### 第7步: 决策与同步

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

---
## 6. 命令参考

*   **生成特征**: `python -m experiment.main gen-feats [--funcs <func_name_1> ...]`
    *   `--funcs`: 指定要生成的一个或多个特征函数。**如果省略，则生成所有非实验性特征**。
    *   `--base-file <filename>`: 指定一个基础特征文件名进行更新，默认为最新。

*   **删除特征**: `python -m experiment.main del-feats --funcs <func_name_1> --cols <col_name_1> ... --base-file <filename>`
    *   `--funcs`: 指定要删除的特征**函数名**。脚本会自动找到该函数生成的所有列并删除它们。
    *   `--cols`: 指定要删除的特征**特征列名**。
    *   `--base-file <filename>`: **必须**指定要操作的基础特征文件名。

*   **模型训练**: `python -m experiment.main train`
    *   `--feature-file <path>`: 指定用于训练的特征文件，默认为最新。
    *   `--save-model`: Flag, 是否保存模型文件。
    *   `--save-oof`: Flag, 是否保存OOF预测文件。
