import json
import os
import pandas as pd

class DataLoader:
    def __init__(self, root_dir, task_type):
        self.root_dir = root_dir
        self.task_type = task_type
        self.data = []

    def load_data(self):
        jsonl_files = self._get_jsonl_files(self.root_dir + self.task_type)
        for file_path in jsonl_files:
            self._read_jsonl(file_path)

    def _get_jsonl_files(self, directory):
        jsonl_files = []
        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith('.jsonl'):
                    jsonl_files.append(os.path.join(dirpath, filename))
        return jsonl_files

    def _read_jsonl(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def get_data(self):
        return self.data

# sample use
if __name__ == "__main__":
    root_directory = 'data/'
    task_type = 'CG/'
    data_loader = DataLoader(root_directory, task_type)
    data_loader.load_data()
    raw_data = data_loader.get_data()