import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取 CSV 文件
df = pd.read_csv('spearman_correlation.csv', index_col=0)  # 第一列作为行索引

# 打印数据以确认格式
print("Correlation Matrix:")
print(df)

# 获取数据的最小值（作为颜色范围的下限）
# vmin = df.min().min()  # 整个矩阵的最小值
vmin = 0.2
vmax = 1  # 上限固定为 1

# 使用 seaborn 创建热图
plt.figure(figsize=(8, 6))
sns.heatmap(
    df,
    annot=True,
    cmap='Blues',  # 使用从深蓝到浅蓝的渐变色
    vmin=vmin,     # 颜色范围下限为数据最小值
    vmax=vmax,     # 颜色范围上限为 1
    linewidths=0.5,
    fmt=".2f"
)

# 设置标题和轴标签
# plt.title('Spearman Correlation Heatmap', fontsize=16, pad=20)  # 增加 pad 参数
# plt.xlabel('Variables', fontsize=14)
# plt.ylabel('Variables', fontsize=14)

# 横轴标签斜着显示
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)  # 纵轴标签不旋转

# 调整布局
# plt.tight_layout()

# 保存热图为 PDF 文件
# plt.savefig('correlation_heatmap.pdf', bbox_inches='tight')
# plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig('correlation_heatmap.pdf', bbox_inches='tight', pad_inches=0.5)
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

# 显示图形
plt.show()