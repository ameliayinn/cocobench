import json


import json



# 从 input.jsonl 读取数据
with open('/data/ywj/cocobench/cocobench-eval/data/CR/CR_old.jsonl', 'r') as infile, open('/data/ywj/cocobench/cocobench-eval/data/CR/CR.jsonl', 'w') as outfile:
    for line in infile:
        # 解析每一行的 JSON 数据
        input_json = json.loads(line)
        input_json1 = input_json["function"]
        
        if not "issue_categories" in input_json1:
            # 准备"issue_categories"结构
            issue_categories = {}
            # 配置已存在的错误类别
            for issue_category in ["syntax_errors", "logical_errors", "performance_issues", "security_issues"]:
                if issue_category in input_json1:
                    issue_categories[issue_category] = input_json1[issue_category]
            
            if issue_categories:
                input_json1["issue_categories"] = issue_categories
        
        # 提升 function 内的每个字段
        output_json = {
            "task_id": input_json["task_id"],
            "description": input_json["function"]["description"],
            "code": input_json["function"]["code"],
            "issue_categories": input_json1["issue_categories"],
            "metadata": input_json["metadata"]
        }
        
        # 将结果写入 output.jsonl
        outfile.write(json.dumps(output_json) + '\n')
