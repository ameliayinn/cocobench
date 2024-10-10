from cocobench.src.util import write_jsonl, read_problems

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)

import typing
import tqdm

BatchGenerator = typing.Callable[
    [PreTrainedModel, PreTrainedTokenizer, str, int], list[str]
]

def eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_samples_per_task: int,
    out_path: str,
    generate_batch_completion: BatchGenerator,
    format_tabs: bool = False,
):
    problems = read_problems()
    samples = []
    pbar = tqdm(total=len(problems) * num_samples_per_task)

    for task_id in problems:
        if format_tabs:
            prompt = problems[task_id]["prompt"].replace("    ", "\t")
        else:
            prompt = problems[task_id]["prompt"]

        batch_completions = generate_batch_completion(
            model, tokenizer, prompt, num_samples_per_task
        )

        for sample in batch_completions:
            result = dict(
                task_id=task_id,
                completion=sample,
            )

            samples += [result]

        pbar.update(num_samples_per_task)

    write_jsonl(out_path, samples)
