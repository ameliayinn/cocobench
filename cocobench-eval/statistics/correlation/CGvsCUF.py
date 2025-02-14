import pandas as pd
import matplotlib.pyplot as plt

# 数据
data = {
    "model": [
        "CodeLlama-7b", "CodeLlama-7b-Instruct", "CodeLlama-13b", "CodeLlama-13b-Instruct",
        "CodeLlama-34b", "CodeLlama-34b-Instruct", "DeepSeek-Coder-1.3b-base", "Deepseek-Coder-1.3b-instruct",
        "DeepSeek-Coder-6.7b-base", "Deepseek-Coder-6.7b-instruct", "DeepSeek-Coder-33b-base", "Deepseek-Coder-33b-instruct",
        "ChatGPT4", "DeepSeek-R1-Distill-Qwen-7B", "o1"
    ],
    "CUF": [12.93, 9.92, 19.01, 13.45, 15.83, 14.88, 14.88, 15.13, 26.05, 35.65, 21.37, 33.88, 53.72, 57.02, 66.12],
    "CG": [11.45, 16.41, 19.23, 18.60, 21.71, 19.08, 15.27, 19.85, 25.78, 46.92, 23.81, 39.53, 44.27, 35.88, 55.73]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(df["CUF"], df["CG"], color='blue', alpha=0.7)

# 添加标题和标签
plt.title('Scatter Plot of CUF vs CG', fontsize=16)
plt.xlabel('CUF', fontsize=14)
plt.ylabel('CG', fontsize=14)

# 添加数据点标签
for i, model in enumerate(df["model"]):
    plt.text(df["CUF"][i], df["CG"][i], model, fontsize=9, ha='right')

# 显示网格
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图形
plt.tight_layout()
plt.savefig('CGvsCUF_scatter.pdf', bbox_inches='tight', pad_inches=0.5)
plt.savefig('CGvsCUF_scatter.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.show()