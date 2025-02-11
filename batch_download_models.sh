#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

'''models=(
  "O1-OPEN/OpenO1-SFT"
)'''

# 读取命令行参数，作为数据集列表
models=("$@")

if [ ${#models[@]} -eq 0 ]; then
  echo "Error: No models provided. Please provide model names as command-line arguments."
  echo "Usage: $0 model1 model2 model3 ..."
  exit 1
fi

download_dir="/data/models/"

hf_token="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"

for model in "${models[@]}"; do
  model_name=$(basename "$model")
  save_path="$download_dir/$model_name"

  echo "Downloading $model..."
  until huggingface-cli download --token $hf_token --local-dir-use-symlinks "False" --resume-download $model --repo-type "model" --local-dir $save_path; do 
    echo "Download failed, retrying..."
  done
  echo "$model downloaded and saved."
  rm -r $save_path/.cache

  echo "success: $model"
done

echo "All models downloaded successfully."
