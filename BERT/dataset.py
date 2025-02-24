import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import json

class SemevalDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        with open(file_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(item["sentence"], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        label = torch.tensor(item["label_id"])
        return {**inputs, "labels": label}

def load_data(train_path, test_path, max_length=128):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = SemevalDataset(train_path, tokenizer, max_length)
    test_dataset = SemevalDataset(test_path, tokenizer, max_length)
    return train_dataset, test_dataset