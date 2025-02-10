import os
import json

# 文件夹路径
folder_path = "/data/ywj/cocobench/cocobench-eval/data_0207"
output_path = "/data/ywj/cocobench/cocobench-eval/data"
if not os.path.exists(output_path):
    os.mkdir(output_path)

# 定义映射规则
difficulty_mapping = {
    "Beginner": "Easy",
    "Intermediate": "Medium",
    "mid": "Medium",
    "Difficult": "Hard",
    "Advanced": "Hard",
    "hard": "Hard",
}

# 遍历文件夹中的所有 JSONL 文件
for dirname in os.listdir(folder_path):
    if dirname.endswith(".py"):
        continue
    dir_path = os.path.join(folder_path, dirname)
    if not os.path.exists(os.path.join(output_path, dirname)) and not dirname.endswith(".py"):
        os.mkdir(os.path.join(output_path, dirname))
    for filename in os.listdir(dir_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(dir_path, filename)
            updated_lines = []

            # 读取文件内容并修改
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    data = json.loads(line)
                    
                    if "metadata" in data and "difficulty" in data["metadata"]:
                        difficulty = data["metadata"]["difficulty"]
                        # 替换难度
                        if difficulty in difficulty_mapping:
                            data["metadata"]["difficulty"] = difficulty_mapping[difficulty]
                    updated_lines.append(json.dumps(data, ensure_ascii=False))
        
            # 写回修改后的内容
            with open(os.path.join(output_path, dirname, filename), "w", encoding="utf-8") as file:
                file.write("\n".join(updated_lines))

print("所有文件已处理完成！")