import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.multiprocessing import Pool, set_start_method
import re

# Optimized prompts for evaluation
PROMPTS = {
    "CG": (
        "You are a comparison expert tasked with determining whether two pieces of code snippet, referred to as 'content' and 'groundtruth', are equivalent."
        "Rules for Evaluation:\n"
        "- Check if the code in content is functional and gives out the right output.\n"
        "- Do not provide explanations, justifications, or additional information in your output.\n\n"
        "Output Requirements:\n"
        "- Respond in a structured JSON format with only the following fields:\n"
        "{\n"
        "  \"task_id\": \"<task_id>\",\n"
        "  \"evaluation\": \"Correct\" or \"Incorrect\"\n"
        "}"
    ),
    "CM": (
        "You are a comparison expert tasked with determining whether two pieces of code snippet, referred to as 'content' and 'groundtruth', are equivalent."
        "Rules for Evaluation:\n"
        "- Check if the code in content is functional and gives out the right output.\n"
        "- Do not provide explanations, justifications, or additional information in your output.\n\n"
        "Output Requirements:\n"
        "- Respond in a structured JSON format with only the following fields:\n"
        "{\n"
        "  \"task_id\": \"<task_id>\",\n"
        "  \"evaluation\": \"Correct\" or \"Incorrect\"\n"
        "}"
    ),
    "CR": (
        "You are a comparison expert tasked with determining whether two pieces of text, referred to as 'content' and 'groundtruth', are equivalent."
        "The text is relevant to errors in code snippets."
        "Rules for Evaluation:\n"
        "- Do not provide explanations, justifications, or additional information in your output.\n\n"
        "Output Requirements:\n"
        "- Respond in a structured JSON format with only the following fields:\n"
        "{\n"
        "  \"task_id\": \"<task_id>\",\n"
        "  \"evaluation\": \"Correct\" or \"Incorrect\"\n"
        "}"
    ),
    "CUF": (
        "You are a comparison expert tasked with determining whether two pieces of text, referred to as 'content' and 'groundtruth', are equivalent."
        "The text is relevant to outputs of code snippets."
        "Rules for Evaluation:\n"
        "- Do not provide explanations, justifications, or additional information in your output.\n\n"
        "Output Requirements:\n"
        "- Respond in a structured JSON format with only the following fields:\n"
        "{\n"
        "  \"task_id\": \"<task_id>\",\n"
        "  \"evaluation\": \"Correct\" or \"Incorrect\"\n"
        "}"
    ),
    "CUR": (
        "You are a comparison expert tasked with determining whether two pieces of text, referred to as 'content' and 'groundtruth', are equivalent."
        "The text is relevant to inputs of code snippets."
        "Rules for Evaluation:\n"
        "- Do not provide explanations, justifications, or additional information in your output.\n\n"
        "Output Requirements:\n"
        "- Respond in a structured JSON format with only the following fields:\n"
        "{\n"
        "  \"task_id\": \"<task_id>\",\n"
        "  \"evaluation\": \"Correct\" or \"Incorrect\"\n"
        "}"
    )
}

def evaluate_sample(model, tokenizer, tasktype, sample, device):
    """Generate an evaluation for a single sample and calculate token usage."""
    try:
        task_id = sample.get("task_id", "unknown")
        content = sample.get("content", "")
        groundtruth = sample.get("groundtruth", "")

        # Append the prompt to the content
        prompt = f"Task ID: {task_id}\n\n{PROMPTS[tasktype]}\n\nContent:\n{content}\n\nGroundtruth:\n{groundtruth}\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.0,  # Disable randomness for deterministic outputs
            do_sample=False,  # Ensure deterministic results
            top_k=50,
            top_p=0.9,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract evaluation using regex
        match = re.search(r'{"task_id":.*}', response)
        if match:
            evaluation = json.loads(match.group(0))
        else:
            evaluation = {"task_id": task_id, "evaluation": "Error: Invalid JSON format"}

        # Count total tokens used
        generated_token_count = outputs.size(1)
        total_token_count = inputs.input_ids.size(1) + generated_token_count

        return {
            "task_id": task_id,
            "evaluation": evaluation,
            "token_count": total_token_count,
        }
    except Exception as e:
        # Handle errors gracefully
        return {
            "task_id": sample.get("task_id", "unknown"),
            "evaluation": f"Error: {str(e)}",
            "token_count": 0,
        }

def process_file(args):
    """Process a single JSONL file."""
    input_path, output_path, model, tokenizer, tasktype, device = args
    results = []

    with open(input_path, "r") as infile:
        for line in infile:
            try:
                sample = json.loads(line)
                evaluation = evaluate_sample(model, tokenizer, tasktype, sample, device)
                results.append(evaluation)
            except json.JSONDecodeError as e:
                print(f"JSON decoding error in file {input_path}: {str(e)}")
            except Exception as e:
                print(f"Unexpected error processing file {input_path}: {str(e)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as outfile:
        for result in results:
            outfile.write(json.dumps(result) + "\n")

def process_all_files(input_dir, results_dir, evaluations_dir, model, tokenizer, device):
    """Process all files in the input directory."""
    for root, _, files in os.walk(os.path.join(results_dir, input_dir)):
        for file in tqdm(files, desc=f"Processing directory: {input_dir}"):
            if file.startswith("result_") and file.endswith(".jsonl"):
                tasktype = next((key for key in PROMPTS.keys() if key in file), None)
                if not tasktype:
                    print(f"Skipping file {file} - unknown task type.")
                    continue

                input_path = os.path.join(root, file)
                output_path = os.path.join(evaluations_dir, input_dir, file.replace("result_", "evaluated_"))

                process_file((input_path, output_path, model, tokenizer, tasktype, device))

def main():
    results_dir = "./cocobench-eval/results"
    evaluations_dir = "./cocobench-eval/evaluations"

    # Get input directories and devices
    input_dirs = ["CodeLlama-7b-hf", "CodeLlama-7b-Instruct-hf", "CodeLlama-13b-hf", "CodeLlama-13b-Instruct-hf", "CodeLlama-34b-hf", "CodeLlama-34b-Instruct-hf"]
    # input_dirs = ["deepseek-coder-1.3b-base", "deepseek-coder-1.3b-instruct", "deepseek-coder-6.7b-base", "deepseek-coder-6.7b-instruct", "deepseek-coder-33b-base", "deepseek-coder-33b-instruct"]
    devices = ["cuda:7", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6"]

    # Load model and tokenizer
    model_path = "/data/models/Meta-Llama-3.1-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    # Use multiprocessing to handle multiple input directories
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    args_list = []
    for input_dir, device in zip(input_dirs, devices):
        args_list.append((input_dir, results_dir, evaluations_dir, model, tokenizer, device))

    with Pool(len(devices)) as pool:
        pool.map(process_all_files, args_list)

if __name__ == "__main__":
    main()
