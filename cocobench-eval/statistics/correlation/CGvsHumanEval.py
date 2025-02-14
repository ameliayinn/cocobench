import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Data
models = [
    "CodeLlama-7b", "CodeLlama-13b", "CodeLlama-34b", "DeepSeek-Coder-1.3b", "DeepSeek-Coder-1.3b-instruct",
    "DeepSeek-Coder-6.7b", "DeepSeek-Coder-33b", "DeepSeek-Coder-6.7b-instruct", "DeepSeek-Coder-33b-instruct",
    "ChatGPT4.0"
]

# Hypothetical values based on the provided image
HumanEval = np.array([45, 50, 52, 30, 35, 40, 55, 65, 78, 85])
CG_pass = np.array([20, 22, 24, 18, 20, 19, 35, 38, 45, 65])

# Create figure
plt.figure(figsize=(10, 6))
sns.regplot(x=HumanEval, y=CG_pass, scatter=True, fit_reg=True, ci=95, 
            scatter_kws={'s': 100, 'alpha': 0.6}, line_kws={'color': 'k', 'linestyle': '--'})

# Add labels to points
for i, model in enumerate(models):
    plt.text(HumanEval[i], CG_pass[i], model, fontsize=9, ha='right', va='bottom')

# Formatting
plt.xlabel("HumanEval@1")
plt.ylabel("CG pass@1")
plt.title("CG pass@1 vs. HumanEval@1")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(["Regression Line", "Models"], loc='upper left')

plt.savefig('CGvsHumanEval.pdf', bbox_inches='tight', pad_inches=0.5)
plt.savefig('CGvsHumanEval.png', dpi=300, bbox_inches='tight', pad_inches=0.5)

plt.show()
