import json

def adjust_and_sort_jsonl_task_ids(jsonl_file_path):
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 解析每一行，提取 task_id 中的数字，并将数字减1
    parsed_lines = []
    for line in lines:
        data = json.loads(line)
        prefix, num = data['task_id'].split('/')
        new_num = int(num) - 1  # 减1操作
        data['task_id'] = f"{prefix}/{new_num}"  # 更新 task_id
        parsed_lines.append((data, new_num))
    
    # 根据调整后的数字进行排序
    sorted_parsed_lines = sorted(parsed_lines, key=lambda x: x[1])
    
    # 把排序后的JSON对象转换回字符串
    sorted_lines = [json.dumps(line[0]) for line in sorted_parsed_lines]

    # 将排序后的数据写回原文件，或者写入新文件
    with open(jsonl_file_path, 'w', encoding='utf-8') as file:  # 覆盖原文件
    #with open(jsonl_file_path.replace('.jsonl', '_adjusted_sorted.jsonl'), 'w', encoding='utf-8') as file:  # 写入新文件
        for line in sorted_lines:
            file.write(line + '\n')

    print("文件处理完成！")

# 提供包含JSONL数据的文件路径
adjust_and_sort_jsonl_task_ids("/data/ywj/cocobench/cocobench-eval/data/CG/CG_cpp.jsonl")