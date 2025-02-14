import numpy as np
from scipy.stats import spearmanr
import pandas as pd

# cd cocobench-eval/statistics/correlation
# python correlation.py > correlation.txt

# 读取CSV数据
data = {
    "model": [
        "CodeLlama-7b", "CodeLlama-7b-Instruct", "CodeLlama-13b", "CodeLlama-13b-Instruct",
        "CodeLlama-34b", "CodeLlama-34b-Instruct", "DeepSeek-Coder-1.3b-base", "Deepseek-Coder-1.3b-instruct",
        "DeepSeek-Coder-6.7b-base", "Deepseek-Coder-6.7b-instruct", "DeepSeek-Coder-33b-base", "Deepseek-Coder-33b-instruct",
        "ChatGPT4", "DeepSeek-R1-Distill-Qwen-7B", "o1"
    ],
    "CUF": [12.93, 9.92, 19.01, 13.45, 15.83, 14.88, 14.88, 15.13, 26.05, 35.65, 21.37, 33.88, 53.72, 57.02, 66.12],
    "CUR": [3.33, 5.17, 4.17, 6.78, 6.67, 4.24, 3.31, 5.98, 5.08, 11.93, 8.40, 10.74, 15.70, 9.09, 9.09],
    "CG": [11.45, 16.41, 19.23, 18.60, 21.71, 19.08, 15.27, 19.85, 25.78, 46.92, 23.81, 39.53, 44.27, 35.88, 55.73],
    "CM": [30.00, 15.00, 20.00, 25.00, 15.00, 20.00, 20.00, 20.00, 25.00, 44.44, 35.00, 50.00, 15.00, 20.00, 45.00],
    "CR": [39.39, 25.71, 32.26, 31.43, 20.00, 29.41, 31.43, 25.71, 28.57, 28.12, 32.35, 37.14, 45.71, 34.29, 45.71],
    # "CoCo-Score": [16.87, 12.94, 16.67, 17.14, 14.50, 15.48, 14.84, 15.62, 19.68, 30.62, 21.91, 31.05, 32.06, 28.26, 39.60]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 为每一列计算排名（1-based）
df_ranked = df.drop('model', axis=1).apply(lambda x: x.rank(method='min', ascending=True)).astype(int)

# 将排名数据与模型名合并
df_ranked['model'] = df['model']

data = pd.DataFrame(df_ranked)
print(data)

# 计算Spearman相关系数
corr, p_value = spearmanr(data.drop('model', axis=1), axis=0)

# 输出相关系数矩阵
print("Spearman Correlation Coefficient Matrix:")
print(corr)
# 将相关系数矩阵保存为 CSV
corr_df = pd.DataFrame(corr, columns=data.columns[:-1], index=data.columns[:-1])
corr_df.to_csv('spearman_correlation.csv')

# 输出p值矩阵
print("P-value Matrix:")
print(p_value)
# 将 p 值矩阵保存为 CSV
p_value_df = pd.DataFrame(p_value, columns=data.columns[:-1], index=data.columns[:-1])
p_value_df.to_csv('spearman_p_value.csv')