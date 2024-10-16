from transformers import (
    AutoTokenizer,
    GPTBigCodeForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)

from core.generation import generate
import os
import torch

TOKEN = "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" # huggingface access token


@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, batch_size: int
) -> list[str]:
    # prompt_input = CG_prompt(prompt)
    input_batch = [prompt for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device) #return pytorch tensors
    input_length = inputs.input_ids.size(dim=1) # get sequence length

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    batch_completions = tokenizer.batch_decode(
        [ids[input_length:] for ids in generated_ids],
        skip_special_tokens=True,
    )

    return batch_completions


if __name__ == "__main__":
    num_samples_per_task = 10
    out_path = "results/wizard/eval.jsonl"
    os.makedirs("results/wizard", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "WizardLM/WizardCoder-15B-V1.0",
        token=TOKEN,
    )

    model = torch.compile(
        GPTBigCodeForCausalLM.from_pretrained(
            "WizardLM/WizardCoder-15B-V1.0",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            max_memory={
                0: "18GiB",
                1: "18GiB",
            },
            use_auth_token=TOKEN,
        ).eval()
    )
    
    tasktype = 'CR'

    generate(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        tasktype,
        generate_batch_completion,
        True,
    )
