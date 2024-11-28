import json

def update_task_id(jsonl_file_path):
    updated_lines = []  # 存储更新后的行

    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析当前行的JSON数据
            data = json.loads(line)
            # 检查并修改task_id字段
            if "task_id" in data:
                task_id = data["task_id"]
                # 此处假设task_id格式为"Cpp/0", "Cpp/1"等，并且希望将大写改为小写
                new_task_id = task_id.lower()
                new_task_id = new_task_id.replace("python/pytorch/", "pytorch/")
                new_task_id = new_task_id.replace("python/tensorflow/", "tensorflow/")
                new_task_id = new_task_id.replace("python/numpy/", "numpy/")
                new_task_id = new_task_id.replace("python/mid/", "python/")
                new_task_id = new_task_id.replace("python/hard/", "python/")
                data["task_id"] = new_task_id
                # 转换回JSON字符串并添加到结果列表
                updated_line = json.dumps(data)
                updated_lines.append(updated_line)

    # 将更新后的数据写回到原文件或新文件
    #with open(jsonl_file_path, 'w', encoding='utf-8') as file:  # 覆盖原文件
    with open(jsonl_file_path.replace('_old.jsonl', '.jsonl'), 'w', encoding='utf-8') as file:  # 或写入新文件
        for line in updated_lines:
            file.write(line + "\n")

    print("文件更新完成！")

# 提供包含JSONL数据的文件路径
'''
update_task_id("/data/ywj/cocobench/cocobench-eval/data/CU/cpp_old.jsonl")
update_task_id("/data/ywj/cocobench/cocobench-eval/data/CU/java_old.jsonl")
update_task_id("/data/ywj/cocobench/cocobench-eval/data/CU/numpy_old.jsonl")
update_task_id("/data/ywj/cocobench/cocobench-eval/data/CU/pytorch_old.jsonl")
update_task_id("/data/ywj/cocobench/cocobench-eval/data/CU/sql_old.jsonl")
update_task_id("/data/ywj/cocobench/cocobench-eval/data/CU/tensorflow_old.jsonl")
'''
update_task_id("/data/ywj/cocobench/cocobench-eval/data/CG/CG_old.jsonl")
'''
update_task_id("/data/ywj/cocobench/cocobench-eval/data/CM/CM.jsonl")
update_task_id("/data/ywj/cocobench/cocobench-eval/data/CR/CR.jsonl")
'''