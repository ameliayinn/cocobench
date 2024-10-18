def extract_code_before_main(code_str):
    lines = code_str.split('\n')
    main_index = -1
    for i in range(len(lines)):
        if "main()" in lines[i]:
            main_index = i
            break
    if main_index == -1:
        return code_str
    else:
        return '\n'.join(lines[:main_index])
    
def remove_empty_lines(code):
    lines = code.split("\n")  # 将代码按行分割成列表
    new_code = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() != "":  # 如果当前行不是空行
            if line.strip().startswith("def ") or line.strip().startswith("class ") or line.strip().startswith("import ") :  # 如果当前行是def函数的定义行
                break  # 停止循环
            new_code += line + "\n"  # 将非空行添加到新的代码字符串中
        i += 1
    new_code += "\n".join(lines[i:])  # 将剩余的代码添加到新的代码字符串中
    return new_code

def process_dataset(sample, problems, test_groundtruth=False):
    task_id = sample["task_id"]
    language = task_id.split("/")[0].lower()
    if test_groundtruth:
        code = sample.get("reference", None)
        if code is None:
            code = sample.get("canonical_solution", None)
            if code is None:
                raise ValueError("can not find reference for sample in test_groundtruth mode", sample)
        sample["generation"] = code
    # supplement reference,prevent indicators that call non-pass@k metrics from being able to obtain reference information
    if sample.get("reference", None) is None:
        if sample.get("canonical_solution", None) is not None:
            sample["reference"] = sample["canonical_solution"]
        else:
            if problems[task_id].get("reference", None) is not None:
                sample["reference"] = problems[task_id]["reference"]
            elif problems[task_id].get("canonical_solution", None) is not None:
                sample["reference"] = problems[task_id]["canonical_solution"]
            else:
                raise ValueError("can not find reference answer for sample", sample)


if __name__ == "__main__":
    