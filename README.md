# 实验一：基于 TF-IDF 的新闻文本分类

仓库链接：https://github.com/Mengqian-coder/AI_project1

本实验面向新闻文本 10 分类任务，使用 TF-IDF 特征比较逻辑回归与多层感知机（MLP）的六组参数配置，并记录验证指标、训练损失和混淆矩阵。根据验证集 Macro-F1 选择最佳配置，再使用全部标注数据重新训练并预测无标签测试集。

## 运行方法

建议使用 Python 3.10 或更高版本。先进入本仓库根目录，再安装依赖：

```bash
git clone https://github.com/Mengqian-coder/AI_project1.git
cd AI_project1
python -m pip install -r requirements.txt
```

课程数据未随仓库发布。请自行获取并把以下两份文件放在 `draft_main.py` 同级目录：

- `train_data.csv`：包含 `text` 和 `target` 两列，共 7,368 条标注文本，标签为 0–9。
- `test_data_unlabeled.csv`：包含 `text` 列，共 2,457 条无标签文本。

数据就位后，在仓库根目录运行：

```bash
python draft_main.py
```

脚本使用相对路径读取数据，运行过程中会重新生成并覆盖下列结果文件。MLP 训练可能需要几分钟，具体取决于计算机性能和早停轮数。

## 实验设置

- 保留标题和正文，删除传输元数据，统一替换邮箱与 URL，并规范化文本。
- 使用固定随机种子 `42` 进行 8:2 分层训练/验证划分。
- TF-IDF 使用词级 unigram、bigram 和至多 30,000 个特征；调参时只在训练子集拟合词表与 IDF。
- 逻辑回归比较 `C=0.1、1、10`；MLP 比较 64/128 单元及两组正则化参数。
- 按验证集 Macro-F1 排序，Accuracy 用于次级排序；测试集不参与模型选择。
- 确定最佳配置后，在全部有标签数据上重新拟合 TF-IDF 并训练最终模型。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `model_results.csv` | 六组候选配置的 Accuracy、Macro-F1 |
| `loss_history.csv` | 调参训练的逐步损失 |
| `logistic_loss_curve.png`、`mlp_loss_curve.png` | 两类模型的训练损失曲线 |
| `best_model_result.csv` | 最佳验证集配置与指标 |
| `best_model_confusion_matrix.csv`、`best_model_confusion_matrix.png` | 按真实类别归一化的验证集混淆矩阵 |
| `validation_errors.csv` | 运行后生成的验证集错分文本；不随仓库发布 |
| `final_training_loss.csv`、`final_model_loss_curve.png` | 全量重新训练的损失记录与曲线 |
| `predictions.csv` | 无标签测试集预测，单列、无表头，保持原始数据顺序 |

逻辑回归曲线记录每次 warm-start 训练后的交叉熵，不含正则项；MLP 曲线来自 `loss_curve_`。两类损失的定义不同，不宜直接比较绝对值。

## 已保存的最佳结果

当前结果中，最佳配置为 **MLP：128 个隐藏单元，`alpha=0.0001`**：

| 验证指标 | 数值 |
| --- | ---: |
| Accuracy | 0.9281 |
| Macro-F1 | 0.9290 |

仓库包含本次实验的指标、图表及 2,457 条测试集预测。重新运行的数值可能受依赖版本及计算环境影响。
