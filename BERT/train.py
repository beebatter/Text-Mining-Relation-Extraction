from dataset import download_data, load_data
import json
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# 读取配置
with open("config/config.json", "r") as f:
    config = json.load(f)

# 确保数据已下载
download_data()

# 加载数据
train_dataset, test_dataset = load_data()

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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

if __name__ == "__main__":
    trainer.train()