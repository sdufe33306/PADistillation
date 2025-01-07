# config.py

import os
from pathlib import Path
import torch


class Config:
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据路径
    labeled_data_dir = Path(r'D:\doc\object_detector_code\PADistillation\data\Knee_X-ray\labeled')  # 有标签数据集路径
    unlabeled_data_dir = Path(r'D:\doc\object_detector_code\PADistillation\data\Knee_X-ray\unlabeled')  # 无标签数据集路径
    validation_data_dir = Path(r'D:\doc\object_detector_code\PADistillation\data\Knee_X-ray\val')  # 验证集数据集路径

    # 超参数
    num_classes = 4  # 根据实际分类数修改
    batch_size = 128
    learning_rate = 1e-3
    num_epochs_pretrain = 50
    num_epochs_semi = 150
    ema_decay = 0.99
    confidence_threshold = 0.9  # 伪标签的置信度阈值
    temperature = 1.0
    alpha = 0.5  # 特征蒸馏损失的权重
    beta = 0.5  # 注意力引导蒸馏损失的权重

    # 早停参数
    early_stopping_patience = 10  # 早停的耐心值（连续多少个epoch验证损失不降低）
    early_stopping_delta = 0.001  # 验证损失的最小变化量

    # 其他配置
    image_size = (160, 160)
    num_workers = 4

        # 保存最佳模型路径
    best_model_path = (r"D:/doc/object_detector_code/PADistillation/saved_models/best_model.pth")
