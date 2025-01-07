# utils.py
import logging
import os
import sys

import torch
from tqdm import tqdm

def update_ema(student_model, teacher_model, ema_decay):
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(ema_decay).add_(student_param.data * (1 - ema_decay))

def save_model(model, path):
    torch.save(model.state_dict(), str(path))

def load_model(model, path):
    model.load_state_dict(torch.load(str(path)))
    return model

def setup_logging(log_file_path):
    """
    设置日志记录配置。

    :param log_file_path: 日志文件的路径
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)

    # 创建控制台处理器
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 添加处理器到日志器
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss = running_loss / len(dataloader.dataset)
    acc = correct / total
    return loss, acc
