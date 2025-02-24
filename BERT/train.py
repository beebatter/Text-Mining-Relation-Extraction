import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from dataset import load_data
import json

# 读取配置文件
with open("config/config.json", "r") as f:
    config = json.load(f)

# 加载数据
train_dataset, test_dataset = load_data(config["train_data_path"], config["test_data_path"], config["max_length"])

# 加载模型
model = BertForSequenceClassification.from_pretrained(config["bert_model"], num_labels=10)

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=config["num_epochs"],
    per_device_train_batch_size=config["batch_size"],
    learning_rate=config["learning_rate"],
    evaluation_strategy="epoch"
)

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 训练
if __name__ == "__main__":
    trainer.train()