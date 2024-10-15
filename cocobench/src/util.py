''' CoCoBench utility functions '''

import os
from typing import *
import json

EVAL_MODEL = {
    "wizard": ["AutoTokenizer", "GPTBigCodeForCausalLM", "PreTrainedTokenizer", "PreTrainedModel"]
}

def write_jsonl(data, output_path):
    """
    write data in a jsonl file
    """
    output_dir = os.path.dirname(output_path)
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_data = []
    for item in data:
        try:
            data = json.dumps(item, ensure_ascii=False)
            write_data.append(data)
        except Exception as e:
            print(f"fail to dumps lines,the line is {item}")
    with open(output_path, "w", encoding='utf-8') as f:
        f.write("\n".join(write_data))