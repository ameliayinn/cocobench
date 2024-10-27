import os
import sys
import yaml
from transformers import AutoTokenizer
from vllm import LLM

from generation import generate

# 加载 YAML 配置文件
def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generate_batch_completion(
    model: LLM, tokenizer: AutoTokenizer, prompt: str, batch_size: int
) -> list[str]:
    input_batch = [prompt] * batch_size
    results = model.generate(input_batch, max_tokens=512, temperature=1, top_p=0.95)

    batch_completions = [result['text'] for result in results]
    return batch_completions


if __name__ == "__main__":
    # 从命令行参数读取 config_path
    if len(sys.argv) < 2:
        print("Error: Please provide the config file path as an argument.")
        print("Usage: python script.py <config_path>")
        sys.exit(1)  # 退出程序并返回错误代码

    config_path = sys.argv[1]  # 获取命令行中传入的配置文件路径
    config_path = os.path.join('./cocobench-eval/config', config_path+'.yaml')

    try:
        # 加载 YAML 文件中的配置
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)  # 退出程序

    # 从 YAML 文件中获取模型路径、输出路径、batch_size 和 max_memory 等配置
    model_path = config['model_path'][0]  # 解析模型路径
    output_path = os.path.join('results', config['output_path'])   # 解析输出路径
    batch_size = config.get('batch_size', 1)  # 如果没有定义 batch_size，默认使用 1
    max_memory = eval(config['max_memory'])   # 解析 max_memory，使用 eval 将字符串转为字典

    # 获取 tasktype 列表
    tasktypes = config['type']  # 解析多个 tasktype

    # num_samples_per_task = 10
    num_samples_per_task = 1

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 加载本地保存的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 使用 vLLM 的 LLMConfig 进行模型加载
    model = LLM(
        model_path,
        gpu_memory_utilization=0.8,  # 设置 GPU 内存利用率
        max_model_len=36576,
    )
    
    # 遍历每个 tasktype，分别调用 generate 函数
    for tasktype in tasktypes:
        # 为每个 tasktype 创建单独的输出目录或文件名
        task_output_path = os.path.join(output_path, f"result_{tasktype}.jsonl")

        print(f"Processing tasktype: {tasktype}, output to: {task_output_path}")

        generate(
            model,
            tokenizer,
            num_samples_per_task,
            task_output_path,  # 使用为该 tasktype 定义的输出路径
            tasktype,          # 当前的 tasktype
            generate_batch_completion,
            False,
        )
