# test.py

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import argparse
import seaborn as sns
from config import Config
from models.backbone import get_model
from utils.utils import evaluate, setup_logging

def test(model, dataloader, criterion, device, logger):
    """
    在测试集上评估模型性能。

    Args:
        model (torch.nn.Module): 训练好的模型。
        dataloader (DataLoader): 测试集数据加载器。
        criterion (torch.nn.Module): 损失函数。
        device (torch.device): 计算设备。
        logger (logging.Logger): 日志记录器。

    Returns:
        tuple: (平均损失, 准确率)
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    avg_acc = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    cm = confusion_matrix(all_labels, all_predictions)

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    logger.info(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('D:/doc/object_detector_code/PADistillation/saved_models/confusion_matrix.png')
    plt.close()
    logger.info("混淆矩阵已保存为 confusion_matrix.png")

    metrics = {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

    return metrics

def main():
    parser = argparse.ArgumentParser(description="测试训练好的模型")
    parser.add_argument('--model_path', type=str, required=True, default=r'D:\doc\object_detector_code\PADistillation\saved_models\best_model.pth')
    parser.add_argument('--test_data_dir', type=str, required=True, default=r'D:\doc\object_detector_code\PADistillation\data\Knee_X-ray\test')
    args = parser.parse_args()

    config = Config()
    config.validation_data_dir = args.test_data_dir  # 使用测试集路径替换验证集路径
    config.best_model_path = args.model_path

    # 设置日志记录
    logger = setup_logging("D:/doc/object_detector_code/PADistillation/saved_models/test.log")
    logger.info("开始测试过程")

    # 数据预处理（与训练时相同）
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 将单通道转换为3通道以适应ResNet
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 使用与预训练模型相同的均值和标准差
                             std=[0.229, 0.224, 0.225]),
    ])

    # 测试集数据集
    test_dataset = datasets.ImageFolder(root=str(config.validation_data_dir), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # 定义模型
    model = get_model(config.num_classes).to(config.device)
    model.load_state_dict(torch.load(config.best_model_path, map_location=config.device))
    model.eval()  # 设置模型为评估模式

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 执行测试
    metrics = test(model, test_loader, criterion, config.device, logger)

    logger.info("测试过程完成。")
    print("测试过程完成。")

if __name__ == "__main__":
    main()
