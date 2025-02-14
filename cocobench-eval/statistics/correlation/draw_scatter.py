import matplotlib.pyplot as plt
import numpy as np

# Data
models = [
    "CodeLlama-7b", "CodeLlama-7b-Instruct", "CodeLlama-13b", "CodeLlama-13b-Instruct",
    "CodeLlama-34b", "CodeLlama-34b-Instruct", "DeepSeek-Coder-1.3b-base", "Deepseek-Coder-1.3b-instruct",
    "DeepSeek-Coder-6.7b-base", "Deepseek-Coder-6.7b-instruct", "DeepSeek-Coder-33b-base", "Deepseek-Coder-33b-instruct",
    "ChatGPT4", "DeepSeek-R1-Distill-Qwen-7B", "o1"
]
CUF = np.array([12.93, 9.92, 19.01, 13.45, 15.83, 14.88, 14.88, 15.13, 26.05, 35.65, 21.37, 33.88, 53.72, 57.02, 66.12])
CG = np.array([11.45, 16.41, 19.23, 18.60, 21.71, 19.08, 15.27, 19.85, 25.78, 46.92, 23.81, 39.53, 44.27, 35.88, 55.73])

# Create figure
plt.figure(figsize=(10, 6))
plt.scatter(CUF, CG, c='b', alpha=0.7, label='Models')

# Add a shaded region
plt.axvspan(min(CUF) - 2, max(CUF) + 2, color='lightgray', alpha=0.3)

# Add regression line
m, b = np.polyfit(CUF, CG, 1)
plt.plot(CUF, m*CUF + b, 'k--', label='Linear Regression')

# Add labels to points
for i, model in enumerate(models):
    plt.text(CUF[i], CG[i], model, fontsize=9, ha='right', va='bottom')

# Formatting
plt.xlabel("CUF")
plt.ylabel("CG")
plt.title("Scatter Plot of CUF vs CG with Grid, Shaded Region, and Regression Line")
plt.grid(True, linestyle='--', alpha=0.6)

plt.legend()
plt.savefig('CGvsCUF_scatter.pdf', bbox_inches='tight', pad_inches=0.5)
plt.savefig('CGvsCUF_scatter.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.show()
