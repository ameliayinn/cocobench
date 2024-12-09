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

# 遍历模型文件夹
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
    length_accuracy[model_type][(model, token_count)].append(accuracy)

# 计算每个模型和长度的平均通过率
final_data = {"Instruct": defaultdict(dict), "Base": defaultdict(dict)}
for model_type, lengths in length_accuracy.items():
    for (model, token_count), accuracies in lengths.items():
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        final_data[model_type][model][token_count] = avg_accuracy

# 获取模型名称
models = [
    "CodeLlama-7b",
    "CodeLlama-13b",
    "CodeLlama-34b",
    "deepsseek-coder-1.3b",
    "deepsseek-coder-6.7b",
    "deepsseek-coder-33b"
]

# 分桶
all_token_counts = list(token_dict.values())
num_buckets = 20
bucket_edges = np.histogram_bin_edges(all_token_counts, bins=num_buckets)
bucket_labels = [
    f"{int(bucket_edges[i])}-{int(bucket_edges[i + 1])}" for i in range(len(bucket_edges) - 1)
]

# 计算每个模型的 bucketed_values 和 smoothed_values
def bucket_values(final_data, models):
    bucketed_values = {"Instruct": defaultdict(list), "Base": defaultdict(list)}
    for model_type in ["Instruct", "Base"]:
        for model in models:
            values = []
            for i in range(len(bucket_edges) - 1):
                bucket_range = (bucket_edges[i], bucket_edges[i + 1])
                bucket_values = [
                    final_data[model_type][model].get(token_count, 0)
                    for token_count in range(int(bucket_range[0]), int(bucket_range[1]))
                ]
                avg_value = sum(bucket_values) / len(bucket_values) if bucket_values else 0
                values.append(avg_value)
            bucketed_values[model_type][model] = values
    return bucketed_values

bucketed_values = bucket_values(final_data, models)

# 移动平均
def moving_average(data, window_size=3):
    if len(data) == 0:
        return np.zeros(len(data))
    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode="same")
    return smoothed

smoothed_values = {
    "Instruct": {model: moving_average(bucketed_values["Instruct"][model]) for model in models},
    "Base": {model: moving_average(bucketed_values["Base"][model]) for model in models}
}

# 绘图：2 行 × 3 列，每张图对比 Base 和 Instruct
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

# 第一个子图：CodeLlama-7b 和 CodeLlama-7b-Instruct
x = np.arange(len(bucket_labels))
ax = axes[0][0]

model_7b = "CodeLlama-7b"
model_7b_instruct = "CodeLlama-7b-Instruct"

base_values = bucketed_values["Base"].get(model_7b, [0] * len(bucket_labels))
instruct_values = bucketed_values["Instruct"].get(model_7b_instruct, [0] * len(bucket_labels))

ax.bar(x - 0.2, base_values, width=0.4, label=f"{model_7b} Base (Buckets)", alpha=0.7)
ax.plot(x, moving_average(base_values), label=f"{model_7b} Base (Smoothed)", color="blue", linestyle="--", marker="o")

ax.bar(x + 0.2, instruct_values, width=0.4, label=f"{model_7b_instruct} Instruct (Buckets)", alpha=0.7)
ax.plot(x, moving_average(instruct_values), label=f"{model_7b_instruct} Instruct (Smoothed)", color="orange", linestyle="--", marker="o")

ax.set_title("CodeLlama-7b Comparison", fontsize=12)
ax.set_xlabel("Token Count Range (Bucket)", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(bucket_labels, rotation=45, ha="right")
ax.legend(fontsize=9)

# 其他子图：显示其他模型
for i, model in enumerate(models[1:]):  # 从第二个模型开始
    row, col = divmod(i + 1, 3)
    ax = axes[row][col]

    base_values = bucketed_values["Base"].get(model, [0] * len(bucket_labels))
    instruct_values = bucketed_values["Instruct"].get(model, [0] * len(bucket_labels))

    ax.bar(x - 0.2, base_values, width=0.4, label=f"{model} Base (Buckets)", alpha=0.7)
    ax.plot(x, moving_average(base_values), label=f"{model} Base (Smoothed)", color="blue", linestyle="--", marker="o")

    ax.bar(x + 0.2, instruct_values, width=0.4, label=f"{model} Instruct (Buckets)", alpha=0.7)
    ax.plot(x, moving_average(instruct_values), label=f"{model} Instruct (Smoothed)", color="orange", linestyle="--", marker="o")

    ax.set_title(f"{model} Comparison", fontsize=12)
    ax.set_xlabel("Token Count Range (Bucket)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, rotation=45, ha="right")
    ax.legend(fontsize=9)

# 添加总标题和调整布局
fig.suptitle("Base vs Instruct Accuracy with Smooth Curves (Evaluations)", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("final.jpg")
plt.show()
