import json


import json



# 从 input.jsonl 读取数据
with open('/data/ywj/cocobench/cocobench-eval/data/CU/CU_sql_old.jsonl', 'r') as infile, open('/data/ywj/cocobench/cocobench-eval/data/CU/CU_sql.jsonl', 'w') as outfile:
    for line in infile:
        # 解析每一行的 JSON 数据
        input_json = json.loads(line)
        output_json = {
            "task_id": input_json["task_id"],
            "description": input_json["description"],
            "code": input_json["code"],
            "input": input_json["table_contents"],
            "expected_output": input_json["expected_output"],
            "metadata": input_json["metadata"]
        }
        
        # 将结果写入 output.jsonl
        outfile.write(json.dumps(output_json) + '\n')
