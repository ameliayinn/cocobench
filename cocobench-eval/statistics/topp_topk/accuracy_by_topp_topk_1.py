import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define the directory containing the evaluation files
base_dir = "../../evaluations"

# Data structure to store results
tasks = set()
results_base = {}
results_instruct = {}

'''
# colors for lines
COLORS = {
    'CG': ''
}
'''

# Parse evaluation files
for model in os.listdir(base_dir):
    model_dir = os.path.join(base_dir, model)
    if not os.path.isdir(model_dir):
        continue

    # Separate base and instruct models
    if "instruct" in model.lower():
        model_dict = results_instruct
        model_name = model.replace('b-Instruct-hf', 'b-instruct')
    elif "b-hf" in model or "base" in model:
        model_dict = results_base
        model_name = model.replace('b-hf', 'b-base')
    else:
        continue
    
    model_dict[model_name] = {}
    
    for file_name in os.listdir(model_dir):
        if file_name.startswith("evaluated_") and file_name.endswith(".jsonl") and file_name.count("_") == 3:
            parts = file_name.split("_")
            if len(parts) == 4:
                task = parts[1]  # Extract task name
                topks = float(parts[2])
                topps = float(parts[3][:3])/100
                
                tasks.add(task)
                if task not in model_dict[model_name]:
                    model_dict[model_name][task] = {}
                
                model_dict[model_name][task][(topps, topks)] = 0

                correct = 0
                total = 0

                with open(os.path.join(model_dir, file_name), "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if data and isinstance(data, dict) and "evaluation" in data:
                                total += 1
                                if data["evaluation"] == "Correct":
                                    correct += 1
                        except json.JSONDecodeError:
                            print(f"Skipping invalid JSON line in {file_name}")

                accuracy = correct / total if total > 0 else 0
                model_dict[model_name][task][(topps, topks)] = accuracy

# Function to plot results
def plot_results(results_base, results_instruct, task_list):
    fig, axes = plt.subplots(2, 5, figsize=(25, 12), dpi=300, sharey=True)
    fig.subplots_adjust(hspace=0.4)  # 增加上下两排图之间的间距
    all_handles_labels = []
    
    for i, task in enumerate(task_list):
        for j, (results, title) in enumerate([(results_base, "Base"), (results_instruct, "Instruct")]):
            ax = axes[j, i]
            
            for model, topp_dict in results.items():
                if task in topp_dict:
                    sorted_keys = sorted(topp_dict[task].keys())
                    sorted_topp = [f"{t[0]:.2f} & {t[1]:.0f}" for t in sorted_keys]
                    sorted_acc = [topp_dict[task][t] for t in sorted_keys]
                    
                    ax.plot(sorted_topp, sorted_acc, linestyle='-', label=model)
                    '''
                    # 允许用户自定义颜色
                    color = COLORS[task]
                    ax.plot(sorted_topp, sorted_acc, linestyle='-', color=color, label=model)
                    '''
                    
                    # Mark the highest point with a pentagram (五角星)
                    max_idx = np.argmax(sorted_acc)
                    ax.plot(sorted_topp[max_idx], sorted_acc[max_idx], marker=(5, 1, 0), markersize=12, color='red')
                    
            ax.set_xticks([f"{t[0]:.2f} & {t[1]:.0f}" for t in sorted_keys if t[0] in {0.75, 0.80, 0.85, 0.90, 0.95}])
            ax.set_title(f"Task: {task} ({title})")
            ax.set_xlabel("top_p & top_k")
            if i == 0:
                ax.set_ylabel("Accuracy")
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Store legend handles and labels
            handles, labels = ax.get_legend_handles_labels()
            all_handles_labels.append((handles, labels))
    
    # Create common legend at the bottom
    handles, labels = zip(*all_handles_labels)
    unique_handles, unique_labels = [], []
    for h, l in zip(sum(handles, []), sum(labels, [])):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
    
    fig.legend(unique_handles, unique_labels, loc='lower center', ncol=6, fontsize=12, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.10, 1, 1])  # 调整布局，留出更多空间
    plt.savefig('all_tasks_topps.pdf', format='pdf', bbox_inches='tight', pad_inches=0.5)
    plt.show()

# Plot all tasks
sorted_tasks = sorted(tasks)[:5]  # Ensure only 5 tasks are plotted
plot_results(results_base, results_instruct, sorted_tasks)
