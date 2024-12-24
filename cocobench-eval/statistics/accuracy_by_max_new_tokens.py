import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define the directory containing the evaluation files
base_dir = "../evaluations"

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
    else:
        model_dict = results_base
    
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

# Plotting

def plot_results(results_base, results_instruct, task_list):
    fig, axes = plt.subplots(2, len(task_list), figsize=(20, 8), sharey=True, dpi=300)
    axes = axes.reshape(2, len(task_list))  # Ensure axes is 2xN for base and instruct models

    for i, task in enumerate(task_list):
        # Plot base models
        ax_base = axes[0, i]
        bar_width = 0.8
        model_names_base = list(results_base.keys())
        all_tokens_base = sorted(set(tokens for model in results_base for tokens in results_base[model].get(task, {})))
        x_base = np.arange(len(all_tokens_base)) * (len(model_names_base) + 1)  # Add spacing

        # Track average
        avg_base = np.zeros(len(all_tokens_base))
        for idx, model in enumerate(model_names_base):
            y = [results_base[model][task].get(tokens, 0) for tokens in all_tokens_base]
            avg_base += np.array(y)
            ax_base.bar(x_base + idx * bar_width, y, bar_width, label=f"{model}", alpha=0.7)
        avg_base /= len(model_names_base) if model_names_base else 1
        ax_base.plot(x_base + bar_width * (len(model_names_base) / 2), avg_base, marker='*', linestyle='--', color='orange', label='Average')

        ax_base.set_xticks(x_base + bar_width * (len(model_names_base) - 1) / 2)
        ax_base.set_xticklabels(all_tokens_base, rotation=45)
        ax_base.set_title(f"Base - Task: {task}")
        ax_base.set_ylabel("Accuracy")
        ax_base.grid(axis='y', linestyle='--', alpha=0.5)
        ax_base.legend(fontsize=6)

        # Plot instruct models
        ax_instruct = axes[1, i]
        model_names_instruct = list(results_instruct.keys())
        all_tokens_instruct = sorted(set(tokens for model in results_instruct for tokens in results_instruct[model].get(task, {})))
        x_instruct = np.arange(len(all_tokens_instruct)) * (len(model_names_instruct) + 1)  # Add spacing

        avg_instruct = np.zeros(len(all_tokens_instruct))
        for idx, model in enumerate(model_names_instruct):
            y = [results_instruct[model][task].get(tokens, 0) for tokens in all_tokens_instruct]
            avg_instruct += np.array(y)
            ax_instruct.bar(x_instruct + idx * bar_width, y, bar_width, label=f"{model}", alpha=0.7)
        avg_instruct /= len(model_names_instruct) if model_names_instruct else 1
        ax_instruct.plot(x_instruct + bar_width * (len(model_names_instruct) / 2), avg_instruct, marker='*', linestyle='--', color='orange', label='Average')

        ax_instruct.set_xticks(x_instruct + bar_width * (len(model_names_instruct) - 1) / 2)
        ax_instruct.set_xticklabels(all_tokens_instruct, rotation=45)
        ax_instruct.set_title(f"Instruct - Task: {task}")
        ax_instruct.set_xlabel("max_new_tokens")
        ax_instruct.set_ylabel("Accuracy")
        ax_instruct.grid(axis='y', linestyle='--', alpha=0.5)
        ax_instruct.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig('img/max_new_tokens.pdf', format='pdf')

# Plot base and instruct models in 2x5 layout
sorted_tasks = sorted(tasks)
plot_results(results_base, results_instruct, sorted_tasks)
