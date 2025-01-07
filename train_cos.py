# train.py
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.utils import update_ema, save_model, evaluate
from utils.hooks import get_feature_maps
from utils.attention import generate_spatial_attention, generate_channel_attention, normalize_attention
from utils.losses import AttentionGuidedDistillationLoss
from utils.EarlyStopping import EarlyStopping  # 导入早停类
import copy

def pretrain(student_model, dataloader, criterion, optimizer, device, num_epochs, config , logger):
    student_model.train()
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(dataloader, desc=f"Pretrain Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = student_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct / total
        print(f"Pretrain Epoch {epoch}/{num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        # 记录日志
        logger.info(f"Pretrain Epoch {epoch}/{num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
    return student_model

def semi_supervised_train(student_model, teacher_model, labeled_loader, unlabeled_loader, val_loader, criterion, optimizer, device, config, logger):
    # 定义需要提取特征图的层名称
    layers = ['layer4']

    # 获取特征图和注册钩子
    teacher_features, teacher_hooks = get_feature_maps(teacher_model, layers)
    student_features, student_hooks = get_feature_maps(student_model, layers)

    # 定义注意力引导蒸馏损失
    attention_distill_loss_fn = AttentionGuidedDistillationLoss(
        temperature=config.temperature,
        alpha=config.alpha,
        beta=config.beta
    )
    # 初始化早停机制
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        delta=config.early_stopping_delta,
        verbose=True,
        path=config.best_model_path
    )

    best_val_acc = 0.0
    for epoch in range(1, config.num_epochs_semi + 1):
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 有标签数据训练
        for inputs, labels in tqdm(labeled_loader, desc=f"Semi Epoch {epoch}/{config.num_epochs_semi} - Labeled"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = student_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 无标签数据生成伪标签
        teacher_model.eval()
        pseudo_labels = []
        pseudo_inputs = []
        with torch.no_grad():
            for inputs, _ in tqdm(unlabeled_loader, desc=f"Semi Epoch {epoch}/{config.num_epochs_semi} - Unlabeled"):
                inputs = inputs.to(device)
                outputs = teacher_model(inputs)
                probs = torch.softmax(outputs, dim=1)
                max_probs, preds = torch.max(probs, dim=1)
                mask = max_probs >= config.confidence_threshold
                pseudo_labels.append(preds[mask])
                pseudo_inputs.append(inputs[mask])

        # 合并所有伪标签和对应的输入
        if len(pseudo_labels) > 0:
            pseudo_labels = torch.cat(pseudo_labels)
            pseudo_inputs = torch.cat(pseudo_inputs)
            if pseudo_labels.numel() > 0:
                student_model.train()
                optimizer.zero_grad()
                outputs = student_model(pseudo_inputs)
                loss_unsup = criterion(outputs, pseudo_labels.to(device))
                loss_unsup.backward()
                optimizer.step()

                running_loss += loss_unsup.item() * pseudo_inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += pseudo_labels.size(0)
                correct += (predicted == pseudo_labels.to(device)).sum().item()

        # 更新教师模型
        update_ema(student_model, teacher_model, config.ema_decay)

        # 特征蒸馏损失计算
        # 使用验证集中的一个批次进行特征提取
        teacher_model.eval()
        student_model.eval()
        try:
            # 从验证集中获取一个批次
            inputs, labels = next(iter(val_loader))
        except StopIteration:
            # 如果验证集为空
            print("验证集为空，无法进行特征蒸馏损失计算。")
            logger.warning("验证集为空，无法进行特征蒸馏损失计算。")
            continue
        inputs = inputs.to(device)

        # 前向传播以提取特征
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
        with torch.enable_grad():
            student_outputs = student_model(inputs)

        # 计算注意力引导蒸馏损失
        distillation_loss = attention_distill_loss_fn(teacher_features, student_features)
        running_loss += distillation_loss.item() * len(layers)  # 假设每层的损失贡献均等

        # 计算epoch损失和准确度
        total_dataset_size = len(labeled_loader.dataset) + (pseudo_labels.numel())
        epoch_loss = running_loss / (total_dataset_size if total_dataset_size > 0 else 1)
        epoch_acc = correct / (total if total > 0 else 1)
        print(f"Semi Epoch {epoch}/{config.num_epochs_semi} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        # 记录日志
        logger.info(f"Semi Epoch {epoch}/{config.num_epochs_semi} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        # Validation
        val_loss, val_acc = evaluate(student_model, val_loader, criterion, device)
        print(f"Semi Epoch {epoch} - Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
        logger.info(f"Semi Epoch {epoch} - Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")

        # 早停判断
        early_stopping(val_loss, student_model, logger)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

        # Check and save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(student_model, config.best_model_path)
            print(f"Semi Epoch {epoch}: Best validation accuracy improved to {val_acc:.4f}. Model saved.")
            logger.info(f"Semi Epoch {epoch}: Best validation accuracy improved to {val_acc:.4f}. Model saved.")

        # 保存每10个epoch的模型
        if epoch % 5 == 0:
            save_model(student_model, f'D:/doc/object_detector_code/PADistillation/saved_models/student_model_epoch_{epoch}.pth')
            save_model(teacher_model, f'D:/doc/object_detector_code/PADistillation/saved_models/teacher_model_epoch_{epoch}.pth')

    # 移除钩子
    for hook in teacher_hooks + student_hooks:
        hook.remove()
    return student_model, teacher_model
