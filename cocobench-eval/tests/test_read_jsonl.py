from typing import Iterable, Dict
import json
import gzip
import os

ROOT = os.path.dirname(os.path.abspath(__file__)) # ./cocobench/tests
COCOBENCH_DATA_DIR = os.path.join(ROOT, "..", "data") # ./cocobench/data

def read_problems(tasktype: str, dirname: str = COCOBENCH_DATA_DIR) -> Dict[str, Dict]:
    task_dir = os.path.join(dirname, tasktype)
    return {task["task_id"]: task for task in read_jsonl(task_dir)}


def read_jsonl(taskdirname: str) -> Iterable[Dict]:
    """
    Reads all .jsonl and .jsonl.gz files in the given directory and yields each line as a dictionary.
    
    Args:
        taskdirname (str): The directory containing .jsonl or .jsonl.gz files.
    
    Yields:
        dict: A dictionary representing each parsed line of the .jsonl files.
    """
    # 遍历指定目录下的所有文件
    for root, _, files in os.walk(taskdirname):
        for file in files:
            # 只处理 .jsonl 和 .jsonl.gz 文件
            if file.endswith(".jsonl") or file.endswith(".jsonl.gz"):
                filepath = os.path.join(root, file)
                
                # 根据文件类型选择合适的读取方式
                if file.endswith(".gz"):
                    with open(filepath, "rb") as gzfp:
                        with gzip.open(gzfp, 'rt') as fp:
                            for line in fp:
                                if any(not x.isspace() for x in line):  # 检查非空行
                                    yield json.loads(line)  # 解析为字典并返回
                else:
                    with open(filepath, "r") as fp:
                        for line in fp:
                            if any(not x.isspace() for x in line):  # 检查非空行
                                yield json.loads(line)  # 解析为字典并返回

if __name__ == "__main__":
    problems = read_problems("CU")
    # print(len(problems))
    print(problems['Java/1'])