from train import trainer
from evaluate import evaluate_model
from dataset import load_data

if __name__ == "__main__":
    # 训练模型
    trainer.train()

    # 评估模型
    _, test_dataset = load_data("data/train.json", "data/test.json", 128)
    evaluate_model(trainer, test_dataset)