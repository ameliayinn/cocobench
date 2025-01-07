import os
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# 定义输入数据文件路径
input_csv = "evaluation_count.csv"

models = [
    "CodeLlama-7b",
    "CodeLlama-13b",
    "CodeLlama-34b",
    "deepseek-coder-1.3b",
    "deepseek-coder-6.7b",
    "deepseek-coder-33b"
]

model_name = {
    "CodeLlama-7b": ["CodeLlama-7b-hf", "CodeLlama-7b-Instruct-hf"],
    "CodeLlama-13b": ["CodeLlama-13b-hf", "CodeLlama-13b-Instruct-hf"],
    "CodeLlama-34b": ["CodeLlama-34b-hf", "CodeLlama-34b-Instruct-hf"],
    "deepseek-coder-1.3b": ["deepseek-coder-1.3b-base", "deepseek-coder-1.3b-instruct"],
    "deepseek-coder-6.7b": ["deepseek-coder-6.7b-base", "deepseek-coder-6.7b-instruct"],
    "deepseek-coder-33b": ["deepseek-coder-33b-base", "deepseek-coder-33b-instruct"]
}

# 读取数据
data = pd.read_csv(input_csv)

# 提取任务类型
data["task_type"] = data["task_id"].apply(lambda x: x.split("/")[0])
task_types = sorted(data["task_type"].unique())

# 数据结构：accuracy_data[task_type][model_type][model][token_count]
accuracy_data = {
    task: {
        "Instruct": {m: defaultdict(list) for m in models},
        "Base": {m: defaultdict(list) for m in models}
    }
    for task in task_types
}

for _, row in data.iterrows():
    model_full = row["model"]
    token_count = row["token_count"]
    correct = row["correct"]
    total = row["total"]
    task_type = row["task_type"]

    accuracy = correct / total if total > 0 else 0
    model_type = "Instruct" if "instruct" in model_full.lower() else "Base"
    
    matched_model = None
    for m in model_name:
        if model_full in model_name[m]:
            matched_model = m
            break
    if matched_model is None:
        continue

    accuracy_data[task_type][model_type][matched_model][token_count].append(accuracy)

# 计算平均准确率
final_data = {
    task: {
        "Instruct": {m: {} for m in models},
        "Base": {m: {} for m in models}
    }
    for task in task_types
}

for task in task_types:
    for mt in ["Instruct", "Base"]:
        for m in models:
            for tc, acc_list in accuracy_data[task][mt][m].items():
                final_data[task][mt][m][tc] = sum(acc_list) / len(acc_list) if acc_list else 0

# 准备绘图
num_tasks_to_plot = min(4, len(task_types))
fig, axes = plt.subplots(2, num_tasks_to_plot, figsize=(20, 10), sharey=True, dpi=300)
fig.suptitle("Base vs Instruct Accuracy by Task and Model with Average Line", fontsize=16)

bar_width = 0.15
num_models = len(models)
offsets = np.linspace(- (num_models - 1) * bar_width / 2, (num_models - 1) * bar_width / 2, num_models)

for i, task in enumerate(task_types[:num_tasks_to_plot]):
    # 对该task的token_count单独分桶
    task_token_counts = []
    for mt in ["Instruct", "Base"]:
        for m in models:
            task_token_counts.extend(list(final_data[task][mt][m].keys()))
    task_token_counts = np.unique(task_token_counts)

    if len(task_token_counts) == 0:
        continue

    task_num_buckets = min(10, len(task_token_counts)) if len(task_token_counts) > 1 else 1
    if task_num_buckets > 1:
        bucket_edges = np.histogram_bin_edges(task_token_counts, bins=task_num_buckets)
    else:
        bucket_edges = [task_token_counts[0]-0.5, task_token_counts[0]+0.5]

    bucket_labels = [
        f"{int(bucket_edges[j])}-{int(bucket_edges[j+1])}"
        for j in range(len(bucket_edges)-1)
    ]

    # 分桶计算
    bucketed_values = {
        "Instruct": {m: [] for m in models},
        "Base": {m: [] for m in models}
    }

    for mt in ["Instruct", "Base"]:
        for m in models:
            for j in range(len(bucket_edges)-1):
                brange = (bucket_edges[j], bucket_edges[j+1])
                vals = [
                    final_data[task][mt][m].get(tc, 0)
                    for tc in task_token_counts
                    if brange[0] <= tc < brange[1]
                ]
                avg_val = sum(vals)/len(vals) if vals else 0
                bucketed_values[mt][m].append(avg_val)

    x = np.arange(len(bucket_labels))

    # 绘制Base行的图表
    ax_base = axes[0, i]
    for model_idx, m in enumerate(models):
        y_values = bucketed_values["Base"][m]
        ax_base.bar(x + offsets[model_idx], y_values, width=bar_width, label=m if i == num_tasks_to_plot-1 else "", alpha=0.7)
    
    # 计算Base行的平均值并绘制折线图
    base_average = []
    for bucket_idx in range(len(bucket_labels)):
        # 该bucket_idx下所有模型的平均
        vals = [bucketed_values["Base"][m][bucket_idx] for m in models]
        bucket_avg = sum(vals)/len(vals) if vals else 0
        base_average.append(bucket_avg)
    ax_base.plot(x, base_average, color="orange", marker="o", linestyle="--", label="Average")

    ax_base.set_title(f"{task} (Base)", fontsize=12)
    ax_base.set_xticks(x)
    ax_base.set_xticklabels(bucket_labels, rotation=45, ha="right")
    ax_base.set_ylim(0, 0.5)
    if i == 0:
        ax_base.set_ylabel("Accuracy")

    # 绘制Instruct行的图表
    ax_instr = axes[1, i]
    for model_idx, m in enumerate(models):
        y_values = bucketed_values["Instruct"][m]
        ax_instr.bar(x + offsets[model_idx], y_values, width=bar_width, label=m if i == num_tasks_to_plot-1 else "", alpha=0.7)
    
    # 计算Instruct行的平均值并绘制折线图
    instr_average = []
    for bucket_idx in range(len(bucket_labels)):
        vals = [bucketed_values["Instruct"][m][bucket_idx] for m in models]
        bucket_avg = sum(vals)/len(vals) if vals else 0
        instr_average.append(bucket_avg)
    ax_instr.plot(x, instr_average, color="orange", marker="o", linestyle="--", label="Average")

    ax_instr.set_title(f"{task} (Instruct)", fontsize=12)
    ax_instr.set_xticks(x)
    ax_instr.set_xticklabels(bucket_labels, rotation=45, ha="right")
    ax_instr.set_ylim(0, 0.5)
    if i == 0:
        ax_instr.set_ylabel("Accuracy")

# 只在最右列的上行子图显示图例
axes[0, num_tasks_to_plot-1].legend(fontsize=9, loc="upper left")

fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("img/Base_vs_Instruct_DynamicBins_with_AverageLine.jpg")
plt.show()
