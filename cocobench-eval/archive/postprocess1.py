import json
import requests

# 设置OpenAI API密钥
API_KEY = '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'  # 替换为你的API密钥
GPT_API_URL = 'https://api.openai.com/v1/chat/completions'

# 读取JSONL文件并处理内容
def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # 解析JSONL格式
            data = json.loads(line)
            content = data['content']
            
            # 构建请求的payload
            prompt = f"Please extract the code snippet and dispose of any other content. {content}"
            response = call_openai_api(prompt)

            # 更新content
            if response:
                data['content'] = response
            
            # 写回JSONL文件
            outfile.write(json.dumps(data) + '\n')

# 调用OpenAI API
def call_openai_api(prompt):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150  # 根据需要调整
    }
    
    response = requests.post(GPT_API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# 运行主程序
if __name__ == '__main__':
    input_file = '/data/ywj/cocobench/results/deepseek-coder-1.3b-base/result_CG.jsonl'  # 输入的JSONL文件名
    output_file = input_file.replace('.jsonl', '_processed.jsonl')  # 输出的JSONL文件名
    process_jsonl(input_file, output_file)
