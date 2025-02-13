import json
import re

with open('/data/ywj/cocobench/cocobench-eval/data_0207/CD/CD.jsonl', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
    for line in lines:
        data = json.load(line)
        question = data['question']
        