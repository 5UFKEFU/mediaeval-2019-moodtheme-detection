#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卷积 + 自注意力混合模型。
"""
import torch, torch.nn as nn, torchvision
from self_attention import AttentionModule

NUM_CLASSES  = 56
HIDDEN_SIZE  = 256

# ---------- MobileNetV2 分支 ----------
class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 灰度 → 3 通道
        self.bw2col = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 10, 1), nn.ReLU(),
            nn.Conv2d(10, 3, 1), nn.ReLU()
        )
        # 轻量级骨干
        self.mv2 = torchvision.models.mobilenet_v2(pretrained=False)
        # 输出层
        self.out_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, HIDDEN_SIZE)
        )
    def forward(self, x):
        x = self.bw2col(x)
        x = self.mv2.features(x)
        x = self.out_conv(x)
        return x

# ---------- 主模型 ----------
class MusicSelfAttModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_branch = MobileNetV2(NUM_CLASSES)
        self.att_branch = nn.Sequential(
            AttentionModule(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        )
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_SIZE, NUM_CLASSES),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入 (B, 96, 1400) →
        x = x.view(-1, 1, 96, 1400)
        feat = self.cnn_branch(x)          # (B, 256)
        att  = feat.view(-1, 16, HIDDEN_SIZE)  # 伪造 seq_len=16
        att  = self.att_branch(att)            # (B, 16, 56)
        att  = torch.mean(att, dim=1)          # Pool
        out  = self.classifier(feat) * att     # 简单融合
        return out

