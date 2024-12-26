import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib

# 设置支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建有向图
G = nx.DiGraph()

# 添加节点和边
G.add_edges_from([
    ("CU (Code Understanding)", "CG (Code Generation)"),
    ("CG (Code Generation)", "bug-free code"),
    ("bug-free code", "CR (Code Review)"),
    ("bug-free code", "CM (Code Modification)"),
    ("CM (Code Modification)", "CG (Code Generation)")
])

# 自定义布局
pos = {
    "CU (Code Understanding)": (0, 2),
    "CG (Code Generation)": (0, 1),
    "bug-free code": (0, 0),
    "CR (Code Review)": (1, -1),
    "CM (Code Modification)": (-1, -1)
}

# 绘制图形
plt.figure(figsize=(20, 6))

# 绘制边
nx.draw_networkx_edges(G, pos, edge_color="black", arrowsize=20)

# 自定义节点样式：圆角矩形
ax = plt.gca()
patches = []
for node, (x, y) in pos.items():
    # 定义圆角矩形
    bbox = FancyBboxPatch(
        (x - 0.15, y - 0.05),  # 左下角坐标
        0.3, 0.1,  # 宽和高
        boxstyle="round,pad=0.2,rounding_size=0.1",  # 圆角样式
        edgecolor="black",
        facecolor="black",
        mutation_aspect=1.0
    )
    patches.append(bbox)
    # 在圆角矩形中添加文字
    ax.text(x, y, node, ha="center", va="center", fontsize=10, color="white")

# 将圆角矩形添加到图中
collection = PatchCollection(patches, match_original=True)
ax.add_collection(collection)

# 设置轴范围和隐藏轴
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 3)
ax.axis("off")

# 保存图片
plt.savefig("flowchart_rounded_rectangles.png", format="png", dpi=300)
plt.show()
