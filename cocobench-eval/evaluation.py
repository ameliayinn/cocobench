import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from collections import Counter

model_path = "/data/models/Llama-3.2-1B-Instruct/"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the SentenceTransformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def process_jsonl(input_file, output_file):
    """Process the JSONL file, calculating similarity scores and writing the results."""
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        lines = infile.readlines()
        # Use tqdm to show progress bar for processing lines
        for line in tqdm(lines, desc=f"Processing {os.path.basename(input_file)}", total=len(lines)):
            data = json.loads(line)
            content = data['content']
            groundtruth = str(data['groundtruth'])
            
            # Skip if content or groundtruth is empty
            if not content or not groundtruth:
                data['similarity_score_cos'] = 0
                data['similarity_score_jaccard'] = 0
                data['similarity_score_rouge'] = 0
            
            # Calculate cosine similarity
            similarity_score_cos = calculate_similarity_cos(content, groundtruth)
            data['similarity_score_cos'] = similarity_score_cos
            
            # Calculate Jaccard similarity
            similarity_score_jaccard = calculate_similarity_jaccard(content, groundtruth)
            data['similarity_score_jaccard'] = similarity_score_jaccard
            
            # Calculate ROUGE similarity
            similarity_score_rouge = calculate_similarity_rouge(content, groundtruth)
            data['similarity_score_rouge'] = similarity_score_rouge
            
            outfile.write(json.dumps(data) + '\n')

def calculate_similarity_cos(content, groundtruth):
    """Calculate cosine similarity between two texts."""
    embeddings1 = embedder.encode(content, convert_to_tensor=True)
    embeddings2 = embedder.encode(groundtruth, convert_to_tensor=True)

    cosine_sim = util.pytorch_cos_sim(embeddings1, embeddings2)  # Range: (-1, 1)
    
    # Convert cosine similarity to a score in the range [0, 100]
    score = (cosine_sim.item() + 1) * 50
    return round(score)

def calculate_similarity_model(content, groundtruth):
    """Calculate similarity using the model."""
    prompt = f"Please evaluate the similarities between the following two texts. Give only a number from 0 to 100.\n\nContent:\n{content}\n\nGround Truth:\n{groundtruth}"
    messages = [{"role": "user", "content": prompt}]
    
    outputs = model(messages, max_new_tokens=10)  # Limit the generated length to 10
    score = outputs[0]["generated_text"].strip()
    
    try:
        score = float(score)  # Attempt to convert output to a float
        return max(0, min(100, round(score)))  # Ensure the score is between 0 and 100
    except ValueError:
        return 0  # Return 0 if parsing fails

def calculate_similarity_jaccard(content, groundtruth):
    """Calculate Jaccard similarity between two texts."""
    set_content = set(content.split())
    set_groundtruth = set(groundtruth.split())
    
    # Calculate Jaccard similarity
    intersection = len(set_content.intersection(set_groundtruth))
    union = len(set_content.union(set_groundtruth))
    
    return (intersection / union * 100) if union > 0 else 0

def calculate_similarity_rouge(content, groundtruth):
    """Calculate ROUGE similarity between two texts."""
    # Tokenize the content and ground truth
    content_tokens = content.split()
    groundtruth_tokens = groundtruth.split()
    
    # Count n-grams (here using ROUGE-1, which is based on words)
    content_counter = Counter(content_tokens)
    groundtruth_counter = Counter(groundtruth_tokens)
    
    intersection = sum((content_counter & groundtruth_counter).values())
    recall = intersection / len(groundtruth_tokens) if len(groundtruth_tokens) > 0 else 0
    precision = intersection / len(content_tokens) if len(content_tokens) > 0 else 0
    
    # Calculate F1 score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    return f1_score * 100

if __name__ == '__main__':
    results_directory = './cocobench-eval/results/'
    dirnames = os.listdir(results_directory)
    # Traverse all files in the results directory
    for dirname in dirnames:
        if dirname.startswith(sys.argv[1]):
            for filename in os.listdir(os.path.join(results_directory, dirname)):
                if filename.startswith("processed_") and filename.endswith(".jsonl"):
                    input_file = os.path.join(results_directory, dirname, filename)
                    output_file = os.path.join(results_directory, dirname, filename.replace("processed_", "evaluated_"))
                    process_jsonl(input_file, output_file)
