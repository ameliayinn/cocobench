import os
import json
import re
import pandas as pd
from collections import defaultdict

# 定义数据路径
evaluations_dir = "../evaluations"
token_csv = "token_count.csv"
output_csv = "evaluation_count.csv"

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
                                    
                                    if not task_id:  # 检查 task_id 是否为空
                                        continue

                                    if task_name in ["CUF", "CUR"]:
                                        task_name = "CU"

                                    full_task_id = f"{task_name}/{task_id}"
                                    token_count = token_dict.get(full_task_id, 0)
                                    
                                    if token_count is None:  # 过滤无效的 token_count
                                        continue

                                    key = (model, full_task_id, token_count)
                                    results_by_model_and_hyperparameters[key]["total"] += 1
                                    if evaluation == "Correct":
                                        results_by_model_and_hyperparameters[key]["correct"] += 1
                                except (json.JSONDecodeError, AttributeError):
                                    continue

# 转换为 DataFrame 并保存为 CSV
rows = []
for (model, full_task_id, token_count), stats in results_by_model_and_hyperparameters.items():
    rows.append({
        "model": model,
        "task_id": full_task_id,
        "token_count": token_count,
        "correct": stats["correct"],
        "total": stats["total"]
    })

df = pd.DataFrame(rows)

# 按照模型和 token_count 排序
df.sort_values(by=["model", "token_count"], inplace=True)

df.to_csv(output_csv, index=False)
print(f"Correct counts saved to {output_csv}")
