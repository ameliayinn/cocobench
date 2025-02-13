import json
import csv
from collections import defaultdict
import re
import os

# run in statistics directionary
evaluations_dir = "../evaluations"

file_pattern = re.compile(r"evaluated_(CG|CM|CR|CUF|CUR)_*.jsonl")

for model in os.listdir(evaluations_dir):
    model_path = os.path.join(evaluations_dir, model)
    
    if os.path.isdir(model_path) and 'o1' in model:
    #if os.path.isdir(model_path):
        with open(f'correctness/correctness_{model}.csv', 'w', newline='') as csv_file:
            fieldnames = ['tasktype', 'correct', 'total', 'rate']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            writer.writeheader()
        
            for file in os.listdir(model_path):
                if file.endswith(".jsonl"):
                    match = file_pattern.match(file)
                    if match:
                        task_name = match.group(1)
                        file_path = os.path.join(model_path, file)
                        
                        # 初始化统计变量
                        correct_count = 0
                        total_count = 0

                        # 读取JSONL文件
                        with open(file_path, 'r') as jsonl_file:
                            for line in jsonl_file:
                                data = json.loads(line)
                                total_count += 1
                                if data['evaluation'] == 'Correct':
                                    correct_count += 1

                        # 计算正确率
                        accuracy = round(correct_count / total_count, 4)

                        # 准备写入CSV的数据
                        csv_data = {
                            'tasktype': task_name,
                            'correct': correct_count,
                            'total': total_count,
                            'rate': accuracy
                        }

                        # 写入CSV文件
                        writer.writerow(csv_data)
                        print("统计结果已保存到 csv 文件中。")