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

    def forward(self, x):
        # ① 把输入统一成 (B,1,96,T)
        if x.dim() == 3:           # (B,96,T)
            x = x.unsqueeze(1)
        elif x.dim() == 4:         # (B,1,96,T)
            pass
        else:
            raise ValueError(f"Bad shape {x.shape}")

        feat = self.backbone(x)                    # (B,256)
        att  = self.att_path(feat.unsqueeze(1).repeat(1,16,1)).mean(1)
        cls  = self.cls_path(feat)
        
        # 确保输出在合理范围内
        att = torch.clamp(att, min=-100, max=100)
        cls = torch.clamp(cls, min=-100, max=100)
        
        return att, cls                           # 两路输出

# --------------------------------------------------
class _MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.to3ch = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 3, 1), nn.ReLU()
        )
        self.net = torchvision.models.mobilenet_v2(weights=None).features
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, HID)
        )

    def forward(self, x):          # x:(B,1,96,T)
        x = self.head(self.net(self.to3ch(x)))
        return x