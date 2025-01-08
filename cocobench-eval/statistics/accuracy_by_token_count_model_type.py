import os
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# 跳转到指定路径
os.chdir("cocobench-eval/statistics")

# 定义输入数据文件路径
input_csv = "evaluation_count.csv"

# 获取模型名称
models = [
    "CodeLlama-7b",
    "CodeLlama-13b",
    "CodeLlama-34b",
    "deepseek-coder-1.3b",
    "deepseek-coder-6.7b",
    "deepseek-coder-33b"
]

# 读取数据
data = pd.read_csv(input_csv)

# 计算每个模型的 Base 和 Instruct 准确率
length_accuracy = defaultdict(lambda: defaultdict(list))

for _, row in data.iterrows():
    model = row["model"]
    token_count = row["token_count"]
    correct = row["correct"]
    total = row["total"]

    accuracy = correct / total if total > 0 else 0
    length_accuracy[model][token_count].append(accuracy)

# 计算每个模型和 token count 的平均准确率
final_data = defaultdict(lambda: defaultdict(list))
for model, lengths in length_accuracy.items():
    for token_count, accuracies in lengths.items():
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        final_data[model][token_count] = avg_accuracy

# 分桶：固定 20 个桶
all_token_counts = data["token_count"].unique()
num_buckets = 12
bucket_edges = np.histogram_bin_edges(all_token_counts, bins=num_buckets)
bucket_labels = [
    f"{int(bucket_edges[i])}-{int(bucket_edges[i + 1])}" for i in range(len(bucket_edges) - 1)
]

# 计算移动平均
def moving_average(data, window_size=3):
    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode="same")
    return smoothed

model_names = {
    "CodeLlama-7b" :["CodeLlama-7b-hf", "CodeLlama-7b-Instruct-hf"],
    "CodeLlama-13b" :["CodeLlama-13b-hf", "CodeLlama-13b-Instruct-hf"],
    "CodeLlama-34b" :["CodeLlama-34b-hf", "CodeLlama-34b-Instruct-hf"],
    "deepseek-coder-1.3b" :["deepseek-coder-1.3b-base", "deepseek-coder-1.3b-instruct"],
    "deepseek-coder-6.7b" :["deepseek-coder-6.7b-base", "deepseek-coder-6.7b-instruct"],
    "deepseek-coder-33b" :["deepseek-coder-33b-base", "deepseek-coder-33b-instruct"]
}

# 将数据分桶
bucketed_values = defaultdict(lambda: {"Instruct": [], "Base": []})
smoothed_values = defaultdict(lambda: {"Instruct": [], "Base": []})

for model in models:
    for _ in range(2):
        model_name = model_names[model][_]
        model_type= "Instruct" if "instruct" in model_name.lower() else "Base"
        for i in range(len(bucket_edges) - 1):
            bucket_range = (bucket_edges[i], bucket_edges[i + 1])
            values = [
                final_data[model_name].get(token_count, 0)
                for token_count in all_token_counts
                if bucket_range[0] <= token_count < bucket_range[1]
            ]
            avg_value = sum(values) / len(values) if values else 0
            bucketed_values[model][model_type].append(avg_value)
        smoothed_values[model][model_type] = moving_average(bucketed_values[model][model_type])

for model in models:
    rows = []
    for i, bucket_label in enumerate(bucket_labels):
        # Base 数据
        rows.append({
            "model": model_names[model][0],
            "bucket": bucket_label,
            "bucketed_value": bucketed_values[model]["Base"][i],
            "smoothed_value": smoothed_values[model]["Base"][i]
        })
        # Instruct 数据
        rows.append({
            "model": model_names[model][1],
            "bucket": bucket_label,
            "bucketed_value": bucketed_values[model]["Instruct"][i],
            "smoothed_value": smoothed_values[model]["Instruct"][i]
        })
    
    # 创建 DataFrame 并保存到 CSV
    model_df = pd.DataFrame(rows)
    output_path = os.path.join("bucketed_values", f"{model}_bucketed_data.csv")
    model_df.to_csv(output_path, index=False)

# 绘图：2 行 × 3 列，每张图对比 Base 和 Instruct
fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharey=True, dpi=300)

for i, model in enumerate(models):
    row, col = divmod(i, 3)
    ax = axes[row][col]

    # Base 模型柱状图和折线图
    x = np.arange(len(bucket_labels))
    ax.bar(
        x - 0.2, bucketed_values[model]["Base"], width=0.4, label=f"{model} Base", alpha=0.7
    )
    ax.plot(
        x, smoothed_values[model]["Base"], label=f"{model} Base", color="blue", linestyle="--", marker="o"
    )

    # Instruct 模型柱状图和折线图
    ax.bar(
        x + 0.2, bucketed_values[model]["Instruct"], width=0.4, label=f"{model} Instruct", alpha=0.7
    )
    ax.plot(
        x, smoothed_values[model]["Instruct"], label=f"{model} Instruct", color="orange", linestyle="--", marker="o"
    )

    ax.set_title(f"{model} Comparison", fontsize=12)
    ax.set_xlabel("Token Count Range (Bucket)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, rotation=45, ha="right")
    ax.legend(fontsize=9)

# 添加总标题和调整布局
fig.suptitle("Base vs Instruct Accuracy with Smooth Curves (2x3 Subplots)", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.95])  # 调整标题位置
plt.savefig("img/Base_vs_Instruct_Comparison_by_Model_Type.pdf", format="pdf")
plt.show()
