import numpy as np
import os
import pandas as pd
import re

# run in statistics directionary

### 选择 TrueSkill 转换方法 ###
def inverse_normalization(task_ratings):
    inverse_mu = {task: 1 / mu for task, mu in task_ratings.items()}
    total_inverse = sum(inverse_mu.values())
    return {task: w / total_inverse for task, w in inverse_mu.items()}

def min_max_scaling(task_ratings):
    mu_max = max(task_ratings.values())
    mu_min = min(task_ratings.values())
    return {task: (mu_max - mu) / (mu_max - mu_min) for task, mu in task_ratings.items()}

def softmax_scaling(task_ratings):
    mu_values = np.array([-mu for mu in task_ratings.values()])  # 负号是为了让小 μ 值有大权重
    exp_values = np.exp(mu_values)
    softmax_weights = exp_values / np.sum(exp_values)
    return {task: softmax_weights[i] for i, task in enumerate(task_ratings)}

# 任务的 TrueSkill 评分 (mu 值) from TrueSkill2.py
task_ratings = {
    "CG": 24.60,
    "CM": 22.43,
    "CR": 25.65, # 最简单
    "CUF": 23.69,
    "CUR": 14.56, # 最难
}

# 选择一种方法来计算权重，最后决定选择inverse_normalization
task_weights = inverse_normalization(task_ratings)  # 可以换成 min_max_scaling 或 softmax_scaling

# 输出权重和最终分数
print(f"任务权重:")
for task, weight in task_weights.items():
    print(f"{task}: {weight:.4f}")

correctness_dir = 'correctness/'
file_names = os.listdir(correctness_dir)
for file_name in file_names:
    if not file_name.endswith('.csv'):
        continue
    file = os.path.join(correctness_dir, file_name)
    df = pd.read_csv(file)
    match = re.search(r'correctness_(.*)\.csv', file_name)
    model_name = match.group(1)
    task_accuracies = dict(zip(df['tasktype'], df['rate']))

    # 计算最终得分
    score = sum(task_accuracies[task] * task_weights[task] for task in task_accuracies)
    print(f"{model_name} 最终得分: {score:.4f}")