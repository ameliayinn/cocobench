from generation import generate
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
model = AutoModelForCausalLM.from_pretrained(
            "/data/models/code_models/codellama/CodeLlama-7b-hf/",  # 从本地路径加载模型
            device_map="auto",
            torch_dtype=torch.bfloat16,
            max_memory = {i: "20GiB" for i in range(8)}
        ).eval()

tokenizer = AutoTokenizer.from_pretrained(
        "/data/models/code_models/codellama/CodeLlama-7b-hf/"  # 从本地路径加载 tokenizer
    )

try:
    generate(
        model=model,
        tokenizer=tokenizer,
        num_samples_per_task=1,
        task_output_path="output.jsonl",
        tasktype="example_task",
        generate_fn=generate_batch_completion,
        debug=False,
        top_k=50  # 假设这是额外的参数
    )
    print("generate 支持 **kwargs")
except TypeError as e:
    print(f"generate 不支持 **kwargs: {e}")