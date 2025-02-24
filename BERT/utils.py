import logging

# 设置日志
logging.basicConfig(filename="logs/training.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def log_info(message):
    logging.info(message)
    print(message)  # 也打印到控制台