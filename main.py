# main.py

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

from config import Config
from models.backbone import get_model
from data.datasets import UnlabeledDataset
from train_cos import pretrain, semi_supervised_train
from utils.utils import save_model
from data.transforms import BlockPixelShuffle
from utils.utils import setup_logging
from utils.hooks import get_feature_maps
from utils.losses import AttentionGuidedDistillationLoss

def main():
    config = Config()
    # 设置日志记录
    logger = setup_logging("D:/doc/object_detector_code/PADistillation/saved_models/training.log")
    logger.info("开始训练过程")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 将单通道转换为3通道以适应ResNet
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 使用与预训练模型相同的均值和标准差
                             std=[0.229, 0.224, 0.225]),
    ])
    # 定义变换管道
    transform1 = transforms.Compose([
        BlockPixelShuffle(block_size=4),  # 添加块内像素洗牌
        transforms.Grayscale(num_output_channels=3),  # 将单通道转换为3通道以适应ResNet
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 使用与预训练模型相同的均值和标准差
                             std=[0.229, 0.224, 0.225]),
    ])

    # 有标签数据集
    labeled_dataset = datasets.ImageFolder(root=config.labeled_data_dir, transform=transform)
    labeled_loader = DataLoader(labeled_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    # 无标签数据集
    unlabeled_dataset = UnlabeledDataset(root=config.unlabeled_data_dir, transform=transform1)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    # 验证集
    validation_dataset = datasets.ImageFolder(root=str(config.validation_data_dir), transform=transform)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False,
                                   num_workers=config.num_workers)

    # 定义模型
    student_model = get_model(config.num_classes).to(config.device)
    teacher_model = get_model(config.num_classes).to(config.device)
    teacher_model.load_state_dict(student_model.state_dict())
    for param in teacher_model.parameters():
        param.requires_grad = False  # 教师模型不需要梯度

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=config.learning_rate)

    # 有监督预训练
    print("开始有监督预训练...")
    student_model = pretrain(
        student_model,
        labeled_loader,
        criterion,
        optimizer,
        config.device,
        config.num_epochs_pretrain,
        config,
        logger
    )
    teacher_model.load_state_dict(student_model.state_dict())
    print("教师模型初始化完成。")

    # 半监督训练
    print("开始半监督训练...")
    student_model, teacher_model = semi_supervised_train(
        student_model,
        teacher_model,
        labeled_loader,
        unlabeled_loader,
        validation_loader,
        criterion,
        optimizer,
        config.device,
        config,
        logger
    )

    # 保存最终模型
    save_model(student_model, os.path.join('D:/doc/object_detector_code/PADistillation/saved_models/student_model_final.pth'))
    save_model(teacher_model, os.path.join('D:/doc/object_detector_code/PADistillation/saved_models/teacher_model_final.pth'))
    print("训练完成。")

if __name__ == "__main__":
    main()
