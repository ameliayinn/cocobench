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
    # Create two separate 1x5 grids
    fig_base, axes_base = plt.subplots(1, 5, figsize=(23, 6), dpi=300, sharey=True)
    # fig_base, axes_base = plt.subplots(1, 5, figsize=(23, 6), dpi=300)
    fig_instruct, axes_instruct = plt.subplots(1, 5, figsize=(25, 6), dpi=300, sharey=True)
    # fig_instruct, axes_instruct = plt.subplots(1, 5, figsize=(25, 6), dpi=300)
    
    # fig_base.subplots_adjust(hspace=0.4)  # Increase vertical space between the tasks in the first grid
    # fig_instruct.subplots_adjust(hspace=0.4)  # Increase vertical space between the tasks in the second grid
    
    all_handles_labels_base = []
    all_handles_labels_instruct = []
    
    for i, task in enumerate(task_list):
        # Plot for the base models
        ax_base = axes_base[i]
        for model, topp_dict in results_base.items():
            if task in topp_dict:
                sorted_keys = sorted(topp_dict[task].keys())
                sorted_topp = [f"{t[0]:.2f} & {t[1]:.0f}" for t in sorted_keys]
                sorted_acc = [topp_dict[task][t] for t in sorted_keys]
                
                ax_base.plot(sorted_topp, sorted_acc, linestyle='-', label=model)
                
                # Mark the highest point with a pentagram (五角星)
                max_idx = np.argmax(sorted_acc)
                ax_base.plot(sorted_topp[max_idx], sorted_acc[max_idx], marker=(5, 1, 0), markersize=9, color='red')
        
        ax_base.set_xticks([f"{t[0]:.2f} & {t[1]:.0f}" for t in sorted_keys if t[0] in {0.75, 0.80, 0.85, 0.90, 0.95}])
        ax_base.set_title(f"Task: {task} (Base)", fontsize=15)
        ax_base.set_xlabel("top_p & top_k", fontsize=15)
        if i == 0:
            ax_base.set_ylabel("Accuracy", fontsize=15)
        ax_base.grid(True, linestyle='--', alpha=0.6)
        
        # Store legend handles and labels
        handles_base, labels_base = ax_base.get_legend_handles_labels()
        all_handles_labels_base.append((handles_base, labels_base))

        # Plot for the instruct models
        ax_instruct = axes_instruct[i]
        for model, topp_dict in results_instruct.items():
            if task in topp_dict:
                sorted_keys = sorted(topp_dict[task].keys())
                sorted_topp = [f"{t[0]:.2f} & {t[1]:.0f}" for t in sorted_keys]
                sorted_acc = [topp_dict[task][t] for t in sorted_keys]
                
                ax_instruct.plot(sorted_topp, sorted_acc, linestyle='-', label=model)
                
                # Mark the highest point with a pentagram (五角星)
                max_idx = np.argmax(sorted_acc)
                ax_instruct.plot(sorted_topp[max_idx], sorted_acc[max_idx], marker=(5, 1, 0), markersize=9, color='red')
        
        ax_instruct.set_xticks([f"{t[0]:.2f} & {t[1]:.0f}" for t in sorted_keys if t[0] in {0.75, 0.80, 0.85, 0.90, 0.95}])
        ax_instruct.set_title(f"Task: {task} (Instruct)", fontsize=15)
        ax_instruct.set_xlabel("top_p & top_k", fontsize=15)
        if i == 0:
            ax_instruct.set_ylabel("Accuracy", fontsize=15)
        ax_instruct.grid(True, linestyle='--', alpha=0.6)
        
        # Store legend handles and labels
        handles_instruct, labels_instruct = ax_instruct.get_legend_handles_labels()
        all_handles_labels_instruct.append((handles_instruct, labels_instruct))
    
    # Create common legend at the bottom for both subplots
    handles_base, labels_base = zip(*all_handles_labels_base)
    unique_handles_base, unique_labels_base = [], []
    for h, l in zip(sum(handles_base, []), sum(labels_base, [])):
        if l not in unique_labels_base:
            unique_labels_base.append(l)
            unique_handles_base.append(h)
    
    # 按字母数字升序排序
    sorted_indices_base = sorted(range(len(unique_labels_base)), key=lambda i: unique_labels_base[i])
    unique_handles_base = [unique_handles_base[i] for i in sorted_indices_base]
    unique_labels_base = [unique_labels_base[i] for i in sorted_indices_base]

    handles_instruct, labels_instruct = zip(*all_handles_labels_instruct)
    unique_handles_instruct, unique_labels_instruct = [], []
    for h, l in zip(sum(handles_instruct, []), sum(labels_instruct, [])):
        if l not in unique_labels_instruct:
            unique_labels_instruct.append(l)
            unique_handles_instruct.append(h)
    
    # 按字母数字升序排序
    sorted_indices_instruct = sorted(range(len(unique_labels_instruct)), key=lambda i: unique_labels_instruct[i])
    unique_handles_instruct = [unique_handles_instruct[i] for i in sorted_indices_instruct]
    unique_labels_instruct = [unique_labels_instruct[i] for i in sorted_indices_instruct]

    fig_base.legend(unique_handles_base, unique_labels_base, loc='lower center', ncol=6, fontsize=15, bbox_to_anchor=(0.5, -0.03))
    fig_instruct.legend(unique_handles_instruct, unique_labels_instruct, loc='lower center', ncol=6, fontsize=15, bbox_to_anchor=(0.5, -0.03))

    fig_base.tight_layout(rect=[0, 0.10, 1, 1])  # Adjust layout for both figures
    fig_instruct.tight_layout(rect=[0, 0.10, 1, 1])  # Adjust layout for both figures
    
    fig_base.savefig('accuracy_by_topp_topk_base.pdf', format='pdf', bbox_inches='tight', pad_inches=0.3)
    fig_instruct.savefig('accuracy_by_topp_topk_instruct.pdf', format='pdf', bbox_inches='tight', pad_inches=0.3)
    
    plt.show()

# Plot all tasks
sorted_tasks = sorted(tasks)[:5]  # Ensure only 5 tasks are plotted
plot_results(results_base, results_instruct, sorted_tasks)