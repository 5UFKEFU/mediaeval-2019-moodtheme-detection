#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileNetV2 -> 256-d 特征
Self-Attention + MLP → 56 标签
兼容 (B,96,T) 旧格式 与 (B,1,96,T) 新格式
"""
import torch, torch.nn as nn, torchvision
from self_attention import AttentionModule

NUM_CLS, HID = 56, 256          # 不动

class MusicSelfAttModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _MobileNetV2()
        self.att_path = nn.Sequential(
            AttentionModule(), nn.Dropout(0.2),
            nn.Linear(HID, NUM_CLS)
        )
        self.cls_path = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(HID, HID),
            nn.Dropout(0.2), nn.Linear(HID, NUM_CLS)
        )
        
        # 添加批归一化层
        self.bn1 = nn.BatchNorm1d(HID)
        self.bn2 = nn.BatchNorm1d(HID)
        
        # 添加额外的特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(HID, HID),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HID, HID),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # Input validation and reshaping
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add channel dimension if missing
        if x.dim() == 3:  # (B,96,T)
            x = x.unsqueeze(1)  # Add channel dimension to make (B,1,96,T)
        elif x.dim() != 4:  # Should be (B,1,96,T)
            raise ValueError(f"Expected input shape (B,96,T) or (B,1,96,T), got {x.shape}")

        # Ensure proper shape for backbone
        if x.size(1) != 1:
            raise ValueError(f"Expected 1 channel, got {x.size(1)} channels")

        # Extract features
        feat = self.backbone(x)  # (B,256)
        
        # Ensure feat is 2D
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        
        # Apply feature extractor
        feat = self.feature_extractor(feat)
        
        # Apply batch normalization
        feat = self.bn1(feat)
        
        # Attention path
        att_feat = feat.unsqueeze(1).repeat(1,16,1)
        att = self.att_path(att_feat).mean(1)
        
        # 确保att是2D的，以便BatchNorm1d可以处理
        if att.dim() == 1:
            att = att.unsqueeze(0)
            
        att = self.bn2(att)
        
        # Classification path
        cls = self.cls_path(feat)
        
        # Ensure outputs are in reasonable range
        att = torch.clamp(att, min=-100, max=100)
        cls = torch.clamp(cls, min=-100, max=100)
        
        return att, cls

# --------------------------------------------------
class _MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.to3ch = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 3, 1), nn.ReLU()
        )
        # 使用预训练的MobileNetV2
        self.net = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1').features
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, HID),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 添加额外的卷积层
        self.extra_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),  # 将输出通道数改为3
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

    def forward(self, x):          # x:(B,1,96,T)
        x = self.to3ch(x)
        x = self.extra_conv(x)
        x = self.net(x)
        x = self.head(x)
        return x