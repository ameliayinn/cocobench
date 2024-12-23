import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 文件路径
input_csv = "evaluation_count.csv"
output_csv_base = "bucketed_values/base_bucketed_values.csv"
output_csv_instruct = "bucketed_values/instruct_bucketed_values.csv"

# 加载数据
data = pd.read_csv(input_csv)

# 计算每个模型类型的准确率
length_accuracy = {"Instruct": defaultdict(list), "Base": defaultdict(list)}
for _, row in data.iterrows():
    model = row["model"]
    token_count = row["token_count"]
    correct = row["correct"]
    total = row["total"]

    accuracy = correct / total if total > 0 else 0
    model_type = "Instruct" if "instruct" in model.lower() else "Base"
    length_accuracy[model_type][token_count].append((model, accuracy))

# 计算每个长度的平均通过率
final_data = {"Instruct": defaultdict(dict), "Base": defaultdict(dict)}
for model_type, lengths in length_accuracy.items():
    for token_count, accuracies in lengths.items():
        for model, accuracy in accuracies:
            final_data[model_type][token_count][model] = accuracy

# 分桶：固定 10 个桶
all_token_counts = data["token_count"].unique()
num_buckets = 8
bucket_edges = np.histogram_bin_edges(all_token_counts, bins=num_buckets)
bucket_labels = [
    f"{int(bucket_edges[i])}-{int(bucket_edges[i + 1])}" for i in range(len(bucket_edges) - 1)
]

bucketed_values = {"Instruct": defaultdict(list), "Base": defaultdict(list)}
models = sorted(data["model"].unique())
for model_type in ["Instruct", "Base"]:
    for i in range(len(bucket_edges) - 1):
        bucket_range = (bucket_edges[i], bucket_edges[i + 1])
        for model in models:
            if (model_type == "Instruct" and "instruct" not in model.lower()) or (
                model_type == "Base" and "instruct" in model.lower()
            ):
                continue
            values = [
                final_data[model_type].get(token_count, {}).get(model, 0)
                for token_count in all_token_counts
                if bucket_range[0] <= token_count < bucket_range[1]
            ]
            avg_value = sum(values) / len(values) if values else 0
            bucketed_values[model_type][model].append(avg_value)

# 移动平均
def moving_average(data, window_size=3):
    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode="same")
    return smoothed

smoothed_values = {
    "Instruct": {model: moving_average(bucketed_values["Instruct"][model]) for model in models if "instruct" in model.lower()},
    "Base": {model: moving_average(bucketed_values["Base"][model]) for model in models if "instruct" not in model.lower()}
}

# 保存数据到 CSV 文件
def save_bucketed_values_to_csv(model_type, bucketed_data, smoothed_data, output_csv):
    rows = []
    for model, values in bucketed_data[model_type].items():
        for i, value in enumerate(values):
            smoothed_value = smoothed_data[model_type][model][i]
            rows.append({"model": model, "bucket": bucket_labels[i], "bucketed_value": value, "smoothed_value": smoothed_value})
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved {model_type} data to {output_csv}")

save_bucketed_values_to_csv("Base", bucketed_values, smoothed_values, output_csv_base)
save_bucketed_values_to_csv("Instruct", bucketed_values, smoothed_values, output_csv_instruct)

# 绘图：Base 和 Instruct 分别画成 1x2 的 subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True, dpi=300)

# 基础设置
x = np.arange(len(bucket_labels))
bar_width = 0.8 / 6  # 每个柱子的宽度
offsets = np.linspace(-2.5 * bar_width / 1, 2.5 * bar_width / 1, 6)  # 模型之间的偏移

# Base 绘图
ax = axes[0]
base_models = [model for model in models if "instruct" not in model.lower()]
for i, model in enumerate(base_models):
    ax.bar(
        x + offsets[i], bucketed_values["Base"][model], width=bar_width,
        label=f"{model} (Buckets)", alpha=0.7
    )
    ax.plot(
        x + offsets[i], smoothed_values["Base"][model],
        label=f"{model} (Smoothed)", linestyle="--", marker="o"
    )
ax.set_title("Base Model Accuracy", fontsize=14)
ax.set_xlabel("Token Count Range (Bucket)", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(bucket_labels, rotation=45, ha="right")
ax.legend(fontsize=9)

# Instruct 绘图
ax = axes[1]
instruct_models = [model for model in models if "instruct" in model.lower()]
for i, model in enumerate(instruct_models):
    ax.bar(
        x + offsets[i], bucketed_values["Instruct"][model], width=bar_width,
        label=f"{model} (Buckets)", alpha=0.7
    )
    ax.plot(
        x + offsets[i], smoothed_values["Instruct"][model],
        label=f"{model} (Smoothed)", linestyle="--", marker="o"
    )
ax.set_title("Instruct Model Accuracy", fontsize=14)
ax.set_xlabel("Token Count Range (Bucket)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(bucket_labels, rotation=45, ha="right")
ax.legend(fontsize=9)

# 添加总标题
fig.suptitle("Base vs Instruct Accuracy by Token Count Buckets", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.95])  # 调整标题位置
plt.savefig("img/Base_vs_Instruct_Comparison_by_Base_Model.pdf", format='pdf')
plt.show()
