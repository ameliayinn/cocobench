import trueskill as ts
import numpy as np
import os
import pandas as pd
import re

# run in statistics directionary

# 初始化 TrueSkill 环境
env = ts.TrueSkill(draw_probability=0.05)  # 设置平局概率，部分任务可能难度相近

# 假设我们有 5 个任务
tasks = ["CG", "CM", "CR", "CUF", "CUR"]
task_ratings = {task: env.create_rating() for task in tasks}  # 初始化任务评分

model_accuracies = {}

correctness_dir = '../correctness/'
file_names = os.listdir(correctness_dir)
for file_name in file_names:
    if not file_name.endswith('.csv'):
        continue
    file = os.path.join(correctness_dir, file_name)
    df = pd.read_csv(file)
    match = re.search(r'correctness_(.*)\.csv', file_name)
    model_name = match.group(1)
    task_accuracies = dict(zip(df['tasktype'], df['rate']))
    model_accuracies[model_name] = task_accuracies
        
print(model_accuracies)
'''
# 假设 3 个模型的准确度数据（示例）
model_accuracies = {
    "Model X": {"Task A": 0.85, "Task B": 0.78, "Task C": 0.92, "Task D": 0.67, "Task E": 0.74},
    "Model Y": {"Task A": 0.80, "Task B": 0.75, "Task C": 0.89, "Task D": 0.70, "Task E": 0.72},
    "Model Z": {"Task A": 0.82, "Task B": 0.77, "Task C": 0.90, "Task D": 0.68, "Task E": 0.73},
}'''

# 生成任务之间的对战（比较任务难度）
matchups = []
for model, accuracies in model_accuracies.items():
    sorted_tasks = sorted(accuracies.keys(), key=lambda t: accuracies[t], reverse=True)  # 按准确度排序
    for i in range(len(sorted_tasks) - 1):
        task_win = sorted_tasks[i]   # 准确率较高的任务（更容易）
        task_lose = sorted_tasks[i+1]  # 准确率较低的任务（更难）
        matchups.append((task_win, task_lose))

# 更新任务 TrueSkill 评分
for winner, loser in matchups:
    task_ratings[winner], task_ratings[loser] = env.rate_1vs1(task_ratings[winner], task_ratings[loser])

# 按难度排序（TrueSkill 分数低 = 更难的任务）
sorted_tasks_by_difficulty = sorted(task_ratings.items(), key=lambda x: x[1].mu)

print("\n任务难度排名（从最难到最容易）:")
for rank, (task, rating) in enumerate(sorted_tasks_by_difficulty, 1):
    print(f"{rank}. {task} (μ={rating.mu:.2f}, σ={rating.sigma:.2f})")