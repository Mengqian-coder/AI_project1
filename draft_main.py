# my_experiment.py

# 1. 从我们提供的帮助脚本中导入加载函数
import pandas as pd

def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled

# 2. 调用函数来获取数据
#    这个函数会自动读取 .csv 文件并返回你需要的所有内容
X_train, y_train, X_test_unlabeled = load_data()

# 3. (验证步骤) 检查一下数据是否加载成功
print("--- 数据加载成功 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 20)

# 打印第一个训练样本和它的标签，感受一下数据
print("第一个训练样本内容:")
print(X_train[0])
print(f"\n第一个训练样本的标签: {y_train[0]}")
print("-" * 20)

# 打印第一个需要你预测的测试样本
print("第一个无标签测试样本内容:")
print(X_test_unlabeled[0])
print("\n" + "="*50)

# --- 在这里开始你的实验！ ---
# 现在，你可以使用 X_train, y_train, 和 X_test_unlabeled 这三个变量
# 来进行后续的特征提取、模型训练和预测了。

# 举例：
# 3. 数据清洗
#清洗函数：保留 Subject 和正文，删除传输元数据并统一文本格式
import re
import unicodedata

def clean_text(text):
    metadata_prefixes = (
        "newsgroups:",
        "path:",
        "message-id:",
        "article-i.d.:",
        "references:",
        "xref:",
        "nntp-posting-host:",
        "organization:",
        "from:",
        "reply-to:",
        "distribution:",
        "lines:",
        "date:",
    )

    def is_metadata_line(line):
        candidate = line.strip()

        # 去掉邮件回复中可能存在的引用符号，例如 >From:
        while candidate.startswith((">", "|", ":")):
            candidate = candidate[1:].lstrip()

        candidate = candidate.lower()
        return candidate.startswith(metadata_prefixes)

    # 老师加载数据时已经转换为字符串，这里再做一次保护。
    if text is None:
        return ""

    text = str(text)

    # 统一 Unicode 字符与换行符。
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 删除不可见控制字符，但保留换行和 Tab。
    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        " ",
        text,
    )

    # Usenet 帖子的头部和正文由第一个空行分隔。
    parts = re.split(
        r"\n[ \t]*\n",
        text,
        maxsplit=1,
    )

    subject = ""
    body = text

    if len(parts) == 2:
        header = parts[0]
        possible_body = parts[1]

        subject_match = re.search(
            r"^Subject[ \t]*:[ \t]*(.*)$",
            header,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        # 只有找到 Subject 时，才将第一部分认定为邮件头，
        # 避免错误删除普通文本的第一段。
        if subject_match is not None:
            subject = subject_match.group(1).strip()
            body = possible_body

    # 删除标题开头可能重复出现的 Re:、FW:、Fwd:。
    subject = re.sub(
        r"^(?:(?:re|fw|fwd)[ \t]*:[ \t]*)+",
        "",
        subject,
        flags=re.IGNORECASE,
    ).strip()

    # 保留引用的讨论内容，但删除其中重复出现的传输元数据。
    body_lines = []

    for line in body.splitlines():
        if not is_metadata_line(line):
            body_lines.append(line)

    body = "\n".join(body_lines)

    # 将标题和正文重新组合。
    cleaned = "\n".join(
        part for part in (subject, body) if part.strip()
    )

    # URL 要在邮箱之前处理，避免带有 @ 的 URL 被邮箱规则破坏。
    cleaned = re.sub(
        r"\b(?:https?://|www\.)[^\s<>]+",
        " URLTOKEN ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 隐藏具体邮箱，防止模型记住作者或机构域名。
    cleaned = re.sub(
        r"(?<![\w.+-])[\w.+-]+@(?:[\w-]+\.)*[\w-]+(?![\w.-])",
        " EMAILTOKEN ",
        cleaned,
    )

    # 合并每一行中的连续空格和 Tab。
    cleaned_lines = []

    for line in cleaned.splitlines():
        line = re.sub(r"[ \t]+", " ", line)
        cleaned_lines.append(line.strip())

    cleaned = "\n".join(cleaned_lines)

    # 最多保留一个空行作为段落分隔。
    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    return cleaned.strip()

#对两个数据集进行清洗            
X_train_clean = [clean_text(text) for text in X_train]
X_test_unlabeled_clean = [clean_text(text) for text in X_test_unlabeled]

# 4.从有标签数据中划分训练集和验证集
from sklearn.model_selection import train_test_split

X_train_texts, X_val_texts, y_train_split, y_val = train_test_split(
    X_train_clean,
    y_train,
    test_size=0.2,#20%有标签数据作为验证集
    random_state=42,#固定划分结果
    stratify=list(y_train),#保持两个子集中各类别比例基本一致
)

print("\n--- 训练集与验证集划分完成 ---")
print("训练集样本数量:", len(X_train_texts))
print("验证集样本数量:", len(X_val_texts))
print("训练集标签数量:", len(y_train_split))
print("验证集标签数量:", len(y_val))

print("\n训练集类别数量:")
print(pd.Series(y_train_split).value_counts().sort_index())

print("\n验证集类别数量:")
print(pd.Series(y_val).value_counts().sort_index())

assert len(X_train_texts) == len(y_train_split)
assert len(X_val_texts) == len(y_val)
assert len(X_train_texts) + len(X_val_texts) == len(X_train_clean)


# 5. 创建TF-IDF向量化器
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",#用Unicode编码方式去除文本中重音特殊音调符号
    ngram_range=(1, 2),#同时提取单个单词和两个单词的组合作为特征
    min_df=2,#至少出现在不同文档才保留特征词
    max_df=0.95,#超过95%的特征词被剔除
    max_features=30000,#最终只保留TF_IDF排名前30000的特征
    sublinear_tf=True,#对词频做做对数缩放
    norm="l2",#输出向量做L2 范数归一化，向量长度等于1
)

# 只使用训练集学习词表和 IDF
X_train_tfidf = vectorizer.fit_transform(X_train_texts)

# 验证集只能使用训练集已经学到的转换规则
X_val_tfidf = vectorizer.transform(X_val_texts)

print("\n--- TF-IDF 特征提取完成 ---")
print("训练集 TF-IDF 形状:", X_train_tfidf.shape)
print("验证集 TF-IDF 形状:", X_val_tfidf.shape)
print("词表大小:", len(vectorizer.vocabulary_))

assert X_train_tfidf.shape[0] == len(X_train_texts)
assert X_val_tfidf.shape[0] == len(X_val_texts)
assert X_train_tfidf.shape[1] == X_val_tfidf.shape[1]

# 6. 定义统一的模型训练与评价函数
from sklearn.metrics import accuracy_score,f1_score

def evaluate_model(
    model_name,
    model,
    X_train_features,
    y_train_labels,
    X_val_features,
    y_val_labels,
    fit_model=True,
):
    if fit_model:
        model.fit(X_train_features, y_train_labels)

    y_val_pred = model.predict(X_val_features)

    accuracy = accuracy_score(y_val_labels, y_val_pred)
    macro_f1 = f1_score(
        y_val_labels,
        y_val_pred,
        average="macro",
        zero_division=0,
    )

    print(f"\n--- {model_name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }

# 7. 训练模型：逻辑回归和MLP
import warnings
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neural_network import MLPClassifier
#逻辑回归逐轮训练和损失记录
def train_logistic_with_history(
    c_value,
    X_features,
    y_labels,
    epochs=60,
):
    model = LogisticRegression(
        C=c_value,
        solver="saga",
        max_iter=1,
        warm_start=True,
        random_state=42,
    )

    loss_history = []

    for epoch in range(1, epochs + 1):
        # max_iter=1 会产生“尚未收敛”的警告，这是逐轮训练时的预期现象。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_features, y_labels)

        train_probabilities = model.predict_proba(X_features)

        current_loss = log_loss(#这里记录的是训练集交叉熵，每次循环后会继承上一轮参数
            y_labels,
            train_probabilities,
            labels=model.classes_,
        )

        loss_history.append(current_loss)

    return model, loss_history

experiment_results = []
loss_records = []
validation_predictions = {}
experiment_configs = {}

#比较逻辑回归的多个参数：正则化参数C
logistic_c_values = [0.1, 1.0, 10.0]

for c_value in logistic_c_values:
    config_name = f"LogisticRegression_C={c_value}"

    logistic_model, logistic_losses = train_logistic_with_history(
        c_value=c_value,
        X_features=X_train_tfidf,
        y_labels=y_train_split,
        epochs=60,
    )

    result = evaluate_model(
        model_name=config_name,
        model=logistic_model,
        X_train_features=X_train_tfidf,
        y_train_labels=y_train_split,
        X_val_features=X_val_tfidf,
        y_val_labels=y_val,
        fit_model=False,
    )

    result["model_type"] = "LogisticRegression"
    result["parameters"] = f"C={c_value}"
    experiment_results.append(result)
    validation_predictions[config_name] = logistic_model.predict(
        X_val_tfidf
    )
    experiment_configs[config_name] = {
        "model_type": "LogisticRegression",
        "C": c_value,
    }

    for step, loss in enumerate(logistic_losses, start=1):
        loss_records.append(
            {
                "model_type": "LogisticRegression",
                "config_name": config_name,
                "step": step,
                "loss": loss,
            }
        )

#比较MLP的多个参数：正则化和隐藏层容量
mlp_configs = [
    {#基础配置
        "hidden_layer_sizes": (64,),
        "alpha": 0.0001,
    },
    {#增强正则化
        "hidden_layer_sizes": (64,),
        "alpha": 0.001,
    },
    {#增大隐藏层容量
        "hidden_layer_sizes": (128,),
        "alpha": 0.0001,
    },
]

for config in mlp_configs:
    hidden_layers = config["hidden_layer_sizes"]
    alpha = config["alpha"]

    config_name = (
        f"MLP_hidden={hidden_layers}_alpha={alpha}"
    )

    mlp_model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        alpha=alpha,
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        batch_size=128,
        max_iter=80,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        random_state=42,
    )

    result = evaluate_model(
        model_name=config_name,
        model=mlp_model,
        X_train_features=X_train_tfidf,
        y_train_labels=y_train_split,
        X_val_features=X_val_tfidf,
        y_val_labels=y_val,
    )

    result["model_type"] = "MLPClassifier"
    result["parameters"] = (
        f"hidden_layer_sizes={hidden_layers}, "
        f"alpha={alpha}"
    )
    experiment_results.append(result)
    validation_predictions[config_name] = mlp_model.predict(
        X_val_tfidf
    )
    experiment_configs[config_name] = {
        "model_type": "MLPClassifier",
        "hidden_layer_sizes": hidden_layers,
        "alpha": alpha,
    }

    for step, loss in enumerate(
        mlp_model.loss_curve_,
        start=1,
    ):
        loss_records.append(
            {
                "model_type": "MLPClassifier",
                "config_name": config_name,
                "step": step,
                "loss": loss,
            }
        )

#保存全部实验结果和损失日志
results_df = pd.DataFrame(experiment_results)

results_df = results_df.sort_values(
    by=["macro_f1", "accuracy"],
    ascending=False,
)

print("\n--- 所有模型验证结果 ---")
print(results_df.to_string(index=False))

results_df.to_csv(
    "model_results.csv",
    index=False,
    encoding="utf-8-sig",
)

loss_df = pd.DataFrame(loss_records)

loss_df.to_csv(
    "loss_history.csv",
    index=False,
    encoding="utf-8-sig",
)

#绘制两个模型的损失曲线
def plot_loss_curves(
    loss_dataframe,
    model_type,
    output_filename,
):
    model_loss_data = loss_dataframe[
        loss_dataframe["model_type"] == model_type
    ]

    plt.figure(figsize=(8, 5))

    for config_name, group in model_loss_data.groupby(
        "config_name",
        sort=False,
    ):
        plt.plot(
            group["step"],
            group["loss"],
            label=config_name,
        )

    plt.xlabel("Training step")
    plt.ylabel("Training loss")
    plt.title(f"{model_type} Training Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_filename,
        dpi=200,
    )
    plt.close()

plot_loss_curves(
    loss_dataframe=loss_df,
    model_type="LogisticRegression",
    output_filename="logistic_loss_curve.png",
)

plot_loss_curves(
    loss_dataframe=loss_df,
    model_type="MLPClassifier",
    output_filename="mlp_loss_curve.png",
)

# 8. 按 Macro-F1 为主、Accuracy 为辅，选择验证集上的最佳方案
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

best_result = results_df.iloc[0]
best_model_name = best_result["model_name"]
best_config = experiment_configs[best_model_name]
best_val_predictions = validation_predictions[best_model_name]

print("\n--- 最佳验证集方案 ---")
print("模型配置:", best_model_name)
print(f"Macro-F1: {best_result['macro_f1']:.4f}")
print(f"Accuracy: {best_result['accuracy']:.4f}")

pd.DataFrame([best_result]).to_csv(
    "best_model_result.csv",
    index=False,
    encoding="utf-8-sig",
)

# 绘制按真实类别归一化的混淆矩阵：行是真实标签，列是预测标签
class_labels = sorted(set(y_train))
normalized_cm = confusion_matrix(
    y_val,
    best_val_predictions,
    labels=class_labels,
    normalize="true",
)

pd.DataFrame(
    normalized_cm,
    index=class_labels,
    columns=class_labels,
).to_csv(
    "best_model_confusion_matrix.csv",
    encoding="utf-8-sig",
)

display = ConfusionMatrixDisplay(
    confusion_matrix=normalized_cm,
    display_labels=class_labels,
)

fig, ax = plt.subplots(figsize=(8, 7))
display.plot(
    ax=ax,
    cmap="Blues",
    values_format=".2f",
    colorbar=False,
)
ax.set_title("Best Model - Validation Confusion Matrix")
plt.tight_layout()
plt.savefig(
    "best_model_confusion_matrix.png",
    dpi=200,
)
plt.close()

# 保存验证集错分样本，便于在实验报告中做错误分析
validation_errors_df = pd.DataFrame(
    {
        "text": X_val_texts,
        "true_label": y_val,
        "predicted_label": best_val_predictions,
    }
)
validation_errors_df = validation_errors_df[
    validation_errors_df["true_label"]
    != validation_errors_df["predicted_label"]
].reset_index(drop=True)

validation_errors_df.to_csv(
    "validation_errors.csv",
    index=False,
    encoding="utf-8-sig",
)

print("验证集错分样本数量:", len(validation_errors_df))

# 9. 最优配置确定后，使用全部有标签数据重新拟合 TF-IDF 和模型
# clone 会复制相同参数，但不会复制之前从训练子集学到的词表和 IDF。
from sklearn.base import clone

final_vectorizer = clone(vectorizer)
X_full_train_tfidf = final_vectorizer.fit_transform(X_train_clean)
X_test_tfidf = final_vectorizer.transform(X_test_unlabeled_clean)

print("\n--- 全量数据 TF-IDF 完成 ---")
print("全量训练集 TF-IDF 形状:", X_full_train_tfidf.shape)
print("测试集 TF-IDF 形状:", X_test_tfidf.shape)

if best_config["model_type"] == "LogisticRegression":
    final_model, final_loss_history = train_logistic_with_history(
        c_value=best_config["C"],
        X_features=X_full_train_tfidf,
        y_labels=y_train,
        epochs=60,
    )
else:
    final_model = MLPClassifier(
        hidden_layer_sizes=best_config["hidden_layer_sizes"],
        alpha=best_config["alpha"],
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        batch_size=128,
        max_iter=80,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        random_state=42,
    )
    final_model.fit(X_full_train_tfidf, list(y_train))
    final_loss_history = list(final_model.loss_curve_)

# 最终全量训练也是一次训练，因此单独保存它的损失记录与曲线
final_loss_df = pd.DataFrame(
    {
        "step": range(1, len(final_loss_history) + 1),
        "loss": final_loss_history,
    }
)
final_loss_df.to_csv(
    "final_training_loss.csv",
    index=False,
    encoding="utf-8-sig",
)

plt.figure(figsize=(8, 5))
plt.plot(
    final_loss_df["step"],
    final_loss_df["loss"],
)
plt.xlabel("Training step")
plt.ylabel("Training loss")
plt.title(f"Final Training Loss - {best_model_name}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    "final_model_loss_curve.png",
    dpi=200,
)
plt.close()

# 10. 对无标签测试集预测，并保存为单列、无表头的提交文件
test_predictions = final_model.predict(X_test_tfidf)
submission_df = pd.DataFrame(test_predictions)

assert submission_df.shape == (len(X_test_unlabeled), 1)
assert not submission_df.isna().any().any()
assert set(test_predictions).issubset(set(y_train))

submission_df.to_csv(
    "predictions.csv",
    index=False,
    header=False,
)

print("\n--- 实验全部完成 ---")
print("最终模型:", best_model_name)
print("测试集预测数量:", len(test_predictions))
print("预测文件: predictions.csv")



