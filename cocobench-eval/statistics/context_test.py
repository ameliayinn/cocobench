import os
import json
import re
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# 定义数据路径
evaluations_dir = "evaluations"
token_csv = "token_statistics.csv"

# 加载 token 数量统计数据
token_stats = pd.read_csv(token_csv)
token_dict = dict(zip(token_stats["task_id"], token_stats["token_count"]))

# 初始化存储结果的字典
results_by_model_and_hyperparameters = defaultdict(lambda: {"correct": 0, "total": 0})
file_pattern = re.compile(r"evaluated_(CG|CM|CR|CUF|CUR)_(\d+)_(\d+).jsonl")

# 遍历每个模型文件夹
for model in os.listdir(evaluations_dir):
    model_path = os.path.join(evaluations_dir, model)
    if os.path.isdir(model_path):
        for file in os.listdir(model_path):
            if file.endswith(".jsonl"):
                match = file_pattern.match(file)
                if match:
                    task_name = match.group(1)
                    file_path = os.path.join(model_path, file)

                    with open(file_path, "r") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    data = json.loads(line.strip())
                                    if data is None:
                                        continue
                                    evaluation = data.get("evaluation")
                                    task_id = data.get("task_id")

                                    if task_name in ["CUF", "CUR"]:
                                        task_name = "CU"

                                    full_task_id = f"{task_name}/{task_id}"
                                    token_count = token_dict.get(full_task_id, 0)

                                    key = (model, task_name, token_count)
                                    results_by_model_and_hyperparameters[key]["total"] += 1
                                    if evaluation == "Correct":
                                        results_by_model_and_hyperparameters[key]["correct"] += 1
                                except (json.JSONDecodeError, AttributeError):
                                    continue

# 聚合数据
length_accuracy = {"Instruct": defaultdict(list), "Base": defaultdict(list)}
for (model, task_name, token_count), stats in results_by_model_and_hyperparameters.items():
    correct = stats["correct"]
    total = stats["total"]
    accuracy = correct / total if total > 0 else 0
    model_type = "Instruct" if "instruct" in model.lower() else "Base"
    length_accuracy[model_type][token_count].append(accuracy)

# 计算每个长度的平均通过率
final_data = {"Instruct": {}, "Base": {}}
for model_type, lengths in length_accuracy.items():
    for token_count, accuracies in lengths.items():
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        final_data[model_type][token_count] = avg_accuracy

# 获取模型名称
models = [
    "CodeLlama-7b",
    "CodeLlama-13b",
    "CodeLlama-34b",
    "deepsseek-coder-1.3b",
    "deepsseek-coder-6.7b",
    "deepsseek-coder-33b"
]

# 分桶：固定 20 个桶
all_token_counts = list(token_dict.values())
num_buckets = 20
bucket_edges = np.histogram_bin_edges(all_token_counts, bins=num_buckets)
bucket_labels = [
    f"{int(bucket_edges[i])}-{int(bucket_edges[i + 1])}" for i in range(len(bucket_edges) - 1)
]

bucketed_values = {"Instruct": defaultdict(list), "Base": defaultdict(list)}
for model_type in ["Instruct", "Base"]:
    for i in range(len(bucket_edges) - 1):
        bucket_range = (bucket_edges[i], bucket_edges[i + 1])
        for model in models:
            values = [
                final_data[model_type].get(token_count, 0)
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
    "Instruct": {model: moving_average(bucketed_values["Instruct"][model]) for model in models},
    "Base": {model: moving_average(bucketed_values["Base"][model]) for model in models}
}

# 绘图：2 行 × 3 列，每张图对比 Base 和 Instruct
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

for i, model in enumerate(models):
    row, col = divmod(i, 3)
    ax = axes[row][col]

    # Base 模型柱状图和折线图
    x = np.arange(len(bucket_labels))
    ax.bar(
        x - 0.2, bucketed_values["Base"][model], width=0.4, label=f"{model} Base (Buckets)", alpha=0.7
    )
    ax.plot(
        x, smoothed_values["Base"][model], label=f"{model} Base (Smoothed)", color="blue", linestyle="--", marker="o"
    )

    # Instruct 模型柱状图和折线图
    ax.bar(
        x + 0.2, bucketed_values["Instruct"][model], width=0.4, label=f"{model} Instruct (Buckets)", alpha=0.7
    )
    ax.plot(
        x, smoothed_values["Instruct"][model], label=f"{model} Instruct (Smoothed)", color="orange", linestyle="--", marker="o"
    )

    ax.set_title(f"{model} Comparison", fontsize=12)
    ax.set_xlabel("Token Count Range (Bucket)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, rotation=45, ha="right")
    ax.legend(fontsize=9)

# 添加总标题和标签
fig.suptitle("Base vs Instruct Accuracy with Smooth Curves (2x3 Subplots)", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.95])  # 调整标题位置
plt.savefig("Base_vs_Instruct_Comparison_with_Curves.jpg")
plt.show()
