import os
import json
import pandas as pd
from datasets import load_dataset
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset

with open("config/config.json", "r") as f:
    config = json.load(f)

def download_data():
    """ 下载 Semeval 2010 Task 8 数据集，并存储为 CSV 格式 """
    os.makedirs("data", exist_ok=True)  # 确保 data 目录存在
    dataset = load_dataset("sem_eval_2010_task_8", download_mode="force_redownload")

    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()

    # 存储到本地 CSV
    df_train.to_csv("data/semeval2010_task8_train.csv", index=False)
    df_test.to_csv("data/semeval2010_task8_test.csv", index=False)
    print("✅ 数据下载完成，存储在 data/ 目录下")

def load_data(train_path=config["train_data_path"], test_path=config["test_data_path"], max_length=128):
    """ 加载 Semeval 2010 Task 8 数据并转换为 PyTorch Dataset """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    class SemevalDataset(Dataset):
        def __init__(self, file_path, tokenizer, max_length=128):
            self.data = pd.read_csv(file_path)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data.iloc[idx]
            inputs = self.tokenizer(item["sentence"], padding="max_length",
                                    truncation=True, max_length=self.max_length,
                                    return_tensors="pt")
            label = torch.tensor(item["label_id"])
            return {**inputs, "labels": label}

    train_dataset = SemevalDataset(train_path, tokenizer, max_length)
    test_dataset = SemevalDataset(test_path, tokenizer, max_length)
    return train_dataset, test_dataset