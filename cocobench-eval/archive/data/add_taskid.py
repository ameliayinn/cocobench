import json

# 读取 input.jsonl 文件
input_file = '/data/node33_backup/dongmh/codebench/data/code_debug_stripped.jsonl'
output_file = '/data/node33_backup/dongmh/codebench/data/code_debug_with_task_id.jsonl'

# 从 JSONL 文件读取数据
with open(input_file, 'r') as jsonl_file:
    input_data = [json.loads(line) for line in jsonl_file]

# 为每个对象添加 task_id
for index, entry in enumerate(input_data):
    # 将 task_id 添加到最前面
    entry_with_task_id = {'task_id': f"Python/{index}", **entry}
    input_data[index] = entry_with_task_id

# 写入新的 JSONL 文件
with open(output_file, 'w') as jsonl_file:
    for entry in input_data:
        jsonl_file.write(json.dumps(entry) + '\n')

print(f"Output written to {output_file}")