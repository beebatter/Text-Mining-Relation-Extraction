from train import trainer
from evaluate import evaluate_model
from dataset import download_data, load_data
import json

with open("config/config.json", "r") as f:
    config = json.load(f)

if __name__ == "__main__":
    # 加载数据
    download_data()

    # 训练模型
    trainer.train()

    # 评估模型
    _, test_dataset = load_data(config["train_data_path"], config["test_data_path"], 128)
    evaluate_model(trainer, test_dataset)