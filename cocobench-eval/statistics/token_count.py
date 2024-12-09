import os
import json
import pandas as pd

# 定义数据目录和任务文件夹
data_dir = "data"
tasks = ["CG", "CM", "CR", "CU"]

# 定义结果存储
results = []

for task in tasks:
    task_dir = os.path.join(data_dir, task)
    if not os.path.isdir(task_dir):
        print(f"Warning: Directory {task_dir} does not exist, skipping.")
        continue

    for file_name in os.listdir(task_dir):
        file_path = os.path.join(task_dir, file_name)
        if file_name.endswith(".jsonl"):  # 检查是否为 JSONL 文件
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    # 逐行读取 JSON 对象
                    for idx, line in enumerate(f):
                        sample = json.loads(line.strip())  # 解析 JSONL 每行内容
                        idx = sample.get('task_id'," ")
                        # 获取任务 ID 和文本内容
                        task_id = f"{task}/{idx}"
                        
                        text = str(sample)# 获取文本内容
                        token_count = len(text)

                        # 添加到结果列表
                        results.append({"task_id": task_id, "token_count": token_count})
                except json.JSONDecodeError as e:
                    print(f"Error reading JSONL file {file_path}: {e}")

# 将结果转换为 DataFrame
df = pd.DataFrame(results)

# 保存到 CSV 文件
output_file = "token_count.csv"
df.to_csv(output_file, index=False)

# 显示表格预览
# import ace_tools as tools; tools.display_dataframe_to_user(name="Token Statistics", dataframe=df)
# print(f"Token statistics saved to {output_file}.")
