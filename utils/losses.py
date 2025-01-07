# utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import generate_spatial_attention, generate_channel_attention, normalize_attention

class AttentionGuidedDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0, alpha=0.5, beta=0.5):
        """
        初始化注意力引导蒸馏损失。

        Args:
            temperature (float): 温度参数用于Softmax归一化。
            alpha (float): 特征蒸馏损失的权重。
            beta (float): 注意力引导蒸馏损失的权重。
        """
        super(AttentionGuidedDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, teacher_features, student_features):
        """
        计算注意力引导蒸馏损失。

        Args:
            teacher_features (dict): 教师模型的特征图字典。
            student_features (dict): 学生模型的特征图字典。

        Returns:
            torch.Tensor: 总损失。
        """
        total_loss = 0.0
        batch_size = next(iter(teacher_features.values())).size(0)  # 获取批量大小

        for layer in teacher_features.keys():
            teacher_feat = teacher_features[layer]  # [B, C, H, W]
            student_feat = student_features[layer]  # [B, C, H, W]

            # 如果特征图尺寸不一致，调整学生特征图尺寸
            if teacher_feat.size()[2:] != student_feat.size()[2:]:
                student_feat = F.interpolate(student_feat, size=teacher_feat.size()[2:], mode='bilinear',
                                             align_corners=False)

            # 生成注意力图
            teacher_spatial_att = generate_spatial_attention(teacher_feat)  # [B, 1, H, W]
            teacher_channel_att = generate_channel_attention(teacher_feat)  # [B, C, 1, 1]

            student_spatial_att = generate_spatial_attention(student_feat)  # [B, 1, H, W]
            student_channel_att = generate_channel_attention(student_feat)  # [B, C, 1, 1]

            # 归一化注意力图
            try:
                teacher_spatial_att_norm = normalize_attention(teacher_spatial_att, self.temperature)  # [B, 1, H*W]
                teacher_channel_att_norm = normalize_attention(teacher_channel_att, self.temperature)  # [B, C, 1]

                student_spatial_att_norm = normalize_attention(student_spatial_att, self.temperature)  # [B, 1, H*W]
                student_channel_att_norm = normalize_attention(student_channel_att, self.temperature)  # [B, C, 1]
            except ValueError as e:
                print(f"Error in layer {layer}: {e}")
                print(f"teacher_spatial_att shape: {teacher_spatial_att.shape}")
                print(f"teacher_channel_att shape: {teacher_channel_att.shape}")
                print(f"student_spatial_att shape: {student_spatial_att.shape}")
                print(f"student_channel_att shape: {student_channel_att.shape}")
                raise e

            # 打印注意力图形状
            print(f"Layer: {layer}")
            print(f"teacher_spatial_att_norm shape: {teacher_spatial_att_norm.shape}")
            print(f"student_spatial_att_norm shape: {student_spatial_att_norm.shape}")

            # 生成注意力掩模
            spatial_mask = (teacher_spatial_att_norm + student_spatial_att_norm) / 2  # [B, 1, H*W]
            channel_mask = (teacher_channel_att_norm + student_channel_att_norm) / 2  # [B, C, 1]

            # 将掩模恢复到原始形状
            B, _, H, W = teacher_feat.size()
            spatial_mask = spatial_mask.view(B, 1, H, W)  # [B, 1, H, W]
            channel_mask = channel_mask.view(B, -1, 1, 1)  # [B, C, 1, 1]

            # 计算特征蒸馏损失
            feature_loss = self.mse_loss(student_feat, teacher_feat)

            # 计算注意力引导蒸馏损失
            spatial_loss = self.mse_loss(student_feat * spatial_mask, teacher_feat * spatial_mask)
            channel_loss = self.mse_loss(student_feat * channel_mask, teacher_feat * channel_mask)

            attention_loss = spatial_loss + channel_loss

            # 总损失
            total_loss += self.alpha * feature_loss + self.beta * attention_loss

        # 平均损失
        total_loss = total_loss / len(teacher_features)

        return total_loss
