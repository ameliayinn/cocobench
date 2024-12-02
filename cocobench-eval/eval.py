import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the evaluation prompt
PROMPT = (
    "You are an evaluation expert tasked with verifying the correctness of algorithmic solutions. "
    "Your responsibilities include analyzing code snippets, validating them against problem descriptions and examples, "
    "and assessing their accuracy based on the expected outputs.\n\n"
    "Your role requires the following:\n"
    "1. Strict Adherence to Input Examples: Ensure the provided `content` code solves all examples as intended in the problem description.\n"
    "2. Comparison with Groundtruth: Validate the `content` code against the `groundtruth` solution to ensure logical correctness and conformity.\n"
    "3. Edge Case Consideration: Check if the `content` code addresses potential edge cases implied by the problem constraints.\n\n"
    "Rules for Evaluation:\n"
    "- Only analyze the code and examples; do not infer or assume unstated requirements.\n"
    "- Do not provide explanations, justifications, or additional information in your output.\n\n"
    "Output Requirements:\n"
    "- Respond in a structured JSON format with only the following fields:\n"
    "{\n"
    "  \"task_id\": \"<task_id>\",\n"
    "  \"evaluation\": \"Correct\" or \"Incorrect\"\n"
    "}"
)

def evaluate_sample(model, tokenizer, sample, device="cuda:0"):
    """Generate an evaluation for a single sample and calculate token usage."""
    try:
        task_id = sample.get("task_id", "unknown")
        content = sample.get("content", "")
        groundtruth = sample.get("groundtruth", "")
        
        # Append the prompt to the content
        prompt = f"Task ID: {task_id}\n\n{PROMPT}\n\nContent:\n{content}\n\nGroundtruth:\n{groundtruth}\n"
        
        # Tokenize and generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        token_count = inputs.input_ids.size(1)  # Count input tokens
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        
        # Decode the model's response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "task_id": task_id,
            "evaluation": response.strip(),
            "token_count": token_count
        }
    except Exception as e:
        # Handle errors and log an evaluation failure
        return {
            "task_id": sample.get("task_id", "unknown"),
            "evaluation": f"Error: {str(e)}",
            "token_count": 0  # Assign 0 token count in case of failure
        }

def process_files(input_dir, output_dir, model, tokenizer, device="cuda:0"):
    """Process all JSONL files in the input directory and save evaluations to the output directory."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jsonl"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Read and process each line in the JSONL file
                with open(input_path, "r") as infile, open(output_path, "w") as outfile:
                    for line in infile:
                        try:
                            sample = json.loads(line)
                            evaluation = evaluate_sample(model, tokenizer, sample, device=device)
                            # Write the evaluation result back to a file
                            outfile.write(json.dumps(evaluation) + "\n")
                        except json.JSONDecodeError as e:
                            # Log errors for JSON parsing issues
                            print(f"Error parsing JSON in file {input_path}: {str(e)}")
                        except Exception as e:
                            # Log other unexpected errors
                            print(f"Unexpected error processing file {input_path}: {str(e)}")

if __name__ == "__main__":
    # Define input and output directories
    input_dir = "./results/CodeLlama-7b-hf"  # Adjust as necessary
    output_dir = "./evaluations/CodeLlama-7b-hf"

    # Load model and tokenizer
    model_path = "/data/models/code_models/deepseek/deepseek-coder-6.7b-instruct"  # Replace with the actual model path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).eval().to("cuda:2")  # Force use of cuda:0

    # Process files
    process_files(input_dir, output_dir, model, tokenizer, device="cuda:2")
