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
    
    model_dict[model] = {}
    
    for file_name in os.listdir(model_dir):
        # Filter files with exactly 4 underscores and len(parts) == 5
        if file_name.startswith("evaluated_") and file_name.endswith(".jsonl") and file_name.count("_") == 4:
            parts = file_name.split("_")
            if len(parts) == 5:
                task = parts[1]  # Extract task name
                max_new_tokens = int(parts[4].split(".")[0])  # Extract max_new_tokens
                
                tasks.add(task)
                if task not in model_dict[model]:
                    model_dict[model][task] = {}

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
                model_dict[model][task][max_new_tokens] = accuracy

# Plotting function
def plot_results(results_base, results_instruct, task_list):
    fig_base, axes_base = plt.subplots(1, len(task_list), figsize=(20, 3), dpi=300, sharey=True)  # ① 保证 Base 图高度与 Instruct 图相同
    fig_instruct, axes_instruct = plt.subplots(1, len(task_list), figsize=(20, 4), dpi=300, sharey=True)  # ① 保持高度一致

    bar_width = 0.8

    if len(task_list) == 1:
        axes_base = [axes_base]
        axes_instruct = [axes_instruct]

    # Plot base models
    for i, task in enumerate(task_list):
        ax_base = axes_base[i]
        model_names_base = list(results_base.keys())
        all_tokens_base = sorted(set(tokens for model in results_base for tokens in results_base[model].get(task, {})))
        x_base = np.arange(len(all_tokens_base)) * (len(model_names_base) + 1)

        avg_base = np.zeros(len(all_tokens_base))
        for idx, model in enumerate(model_names_base):
            y = [results_base[model][task].get(tokens, 0) for tokens in all_tokens_base]
            avg_base += np.array(y)
            ax_base.bar(x_base + idx * bar_width, y, bar_width, label=model, alpha=0.7)
        avg_base /= len(model_names_base) if model_names_base else 1
        ax_base.plot(x_base + bar_width * (len(model_names_base) / 2), avg_base, marker='*', linestyle='--', color='orange', label='Average')

        ax_base.set_xticks(x_base + bar_width * (len(model_names_base) - 1) / 2)
        ax_base.set_xticklabels(all_tokens_base, rotation=45, fontsize=10)  # ③ 统一字体大小
        ax_base.set_title(f"Task: {task} (Base)", fontsize=10)
        ax_base.set_ylabel("Accuracy", fontsize=10)  # ③ 统一字体大小
        ax_base.set_xlabel("max_new_tokens", fontsize=10)  # ③ 统一字体
        ax_base.grid(axis='y', linestyle='--', alpha=0.5)

    # Plot instruct models
    for i, task in enumerate(task_list):
        ax_instruct = axes_instruct[i]
        model_names_instruct = list(results_instruct.keys())
        all_tokens_instruct = sorted(set(tokens for model in results_instruct for tokens in results_instruct[model].get(task, {})))
        x_instruct = np.arange(len(all_tokens_instruct)) * (len(model_names_instruct) + 1)

        avg_instruct = np.zeros(len(all_tokens_instruct))
        for idx, model in enumerate(model_names_instruct):
            y = [results_instruct[model][task].get(tokens, 0) for tokens in all_tokens_instruct]
            avg_instruct += np.array(y)
            ax_instruct.bar(x_instruct + idx * bar_width, y, bar_width, label=model, alpha=0.7)
        avg_instruct /= len(model_names_instruct) if model_names_instruct else 1
        ax_instruct.plot(x_instruct + bar_width * (len(model_names_instruct) / 2), avg_instruct, marker='*', linestyle='--', color='orange', label='Average')

        ax_instruct.set_xticks(x_instruct + bar_width * (len(model_names_instruct) - 1) / 2)
        ax_instruct.set_xticklabels(all_tokens_instruct, rotation=45, fontsize=12)  # ③ 统一字体大小
        ax_instruct.set_title(f"Task: {task} (Instruct)", fontsize=12)
        ax_instruct.set_xlabel("max_new_tokens", fontsize=12)  # ③ 统一字体大小
        ax_instruct.set_ylabel("Accuracy", fontsize=12)  # ③ 统一字体大小
        ax_instruct.grid(axis='y', linestyle='--', alpha=0.5)

    # ② 统一图例放到底部
    '''
    # 获取 Base 任务的图例
    handles_base, labels_base = axes_base[0].get_legend_handles_labels()
    fig_base.legend(handles_base, labels_base, loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=15, ncol=4)

    # 获取 Instruct 任务的图例
    handles_instruct, labels_instruct = axes_instruct[0].get_legend_handles_labels()
    fig_instruct.legend(handles_instruct, labels_instruct, loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=15, ncol=4)
    
    # fig_base.legend(model_names_base + ["Average"], loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=12, ncol=7)  
    # fig_instruct.legend(model_names_instruct + ["Average"], loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=12, ncol=7)
    '''
    
    # 获取 Base 任务的图例
    handles_base, labels_base = axes_base[0].get_legend_handles_labels()

    # 按字母顺序排序（排除 "Average"）
    sorted_indices = sorted(range(len(labels_base)), key=lambda i: labels_base[i].lower() if labels_base[i] != "Average" else "zzz")
    handles_base = [handles_base[i] for i in sorted_indices]
    labels_base = [labels_base[i] for i in sorted_indices]

    # 把 "Average" 放到最后
    if "Average" in labels_base:
        avg_idx = labels_base.index("Average")
        handles_base.append(handles_base.pop(avg_idx))
        labels_base.append(labels_base.pop(avg_idx))

    # 设置图例，ncol=3，并让 "Average" 独立在右侧
    fig_base.legend(handles_base, labels_base, loc='lower center', bbox_to_anchor=(0.5, -0.4), fontsize=13, ncol=4, columnspacing=1.5)

    # 对 instruct 任务的图例做相同处理
    handles_instruct, labels_instruct = axes_instruct[0].get_legend_handles_labels()
    sorted_indices = sorted(range(len(labels_instruct)), key=lambda i: labels_instruct[i].lower() if labels_instruct[i] != "Average" else "zzz")
    handles_instruct = [handles_instruct[i] for i in sorted_indices]
    labels_instruct = [labels_instruct[i] for i in sorted_indices]

    if "Average" in labels_instruct:
        avg_idx = labels_instruct.index("Average")
        handles_instruct.append(handles_instruct.pop(avg_idx))
        labels_instruct.append(labels_instruct.pop(avg_idx))

    fig_instruct.legend(handles_instruct, labels_instruct, loc='lower center', bbox_to_anchor=(0.5, -0.25), fontsize=15, ncol=4, columnspacing=1.5)
    
    plt.tight_layout()

    # Save separate figures for Base and Instruct models
    fig_base.savefig('max_new_tokens_base.pdf', format='pdf', bbox_inches='tight', pad_inches=0.3)
    fig_instruct.savefig('max_new_tokens_instruct.pdf', format='pdf', bbox_inches='tight', pad_inches=0.3)

    plt.show()

# Plot base and instruct models in 1x5 layout
sorted_tasks = sorted(tasks)[:5]  
plot_results(results_base, results_instruct, sorted_tasks)