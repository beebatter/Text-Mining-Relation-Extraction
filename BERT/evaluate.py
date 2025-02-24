import torch
from sklearn.metrics import classification_report
from transformers import Trainer
from dataset import load_data

def evaluate_model(trainer: Trainer, test_dataset):
    """ 评估模型并打印 F1-score """
    preds = trainer.predict(test_dataset).predictions.argmax(axis=1)
    labels = [d["labels"].item() for d in test_dataset]
    print(classification_report(labels, preds, digits=4))

if __name__ == "__main__":
    from train import trainer
    _, test_dataset = load_data("data/train.json", "data/test.json", 128)
    evaluate_model(trainer, test_dataset)