#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
入口脚本：负责准备数据集 DataLoader、启动 Solver 训练 / 验证 / 测试。
兼容 CUDA 12 + CentOS 8。
"""
import os
import torch.multiprocessing as mp
from dataloader import get_audio_loader
from solver import Solver

# --------------------------- 路径配置 ---------------------------
PATH = "../data"
DATA_PATH = f"{PATH}/mediaeval-2019-jamendo/"
LABELS_TXT = f"{PATH}/moodtheme_split.txt"
TRAIN_PATH = f"{PATH}/autotagging_moodtheme-train.tsv"
VAL_PATH   = f"{PATH}/autotagging_moodtheme-validation.tsv"
TEST_PATH  = f"{PATH}/autotagging_moodtheme-test.tsv"

CONFIG = {
    "log_dir":  "./output",
    "batch_size": 8,
}

# --------------------------- 辅助函数 ---------------------------
def get_labels_to_idx(labels_txt):
    labels_to_idx, tag_list = {}, []
    with open(labels_txt) as f:
        for i, line in enumerate(f):
            label = line.strip()
            labels_to_idx[label] = i
            tag_list.append(label)
    return labels_to_idx, tag_list

def train():
    labels_to_idx, tag_list = get_labels_to_idx(LABELS_TXT)

    train_loader1 = get_audio_loader(DATA_PATH, TRAIN_PATH, labels_to_idx,
                                     batch_size=CONFIG["batch_size"])
    train_loader2 = get_audio_loader(DATA_PATH, TRAIN_PATH, labels_to_idx,
                                     batch_size=CONFIG["batch_size"])
    val_loader    = get_audio_loader(DATA_PATH, VAL_PATH,  labels_to_idx,
                                     batch_size=CONFIG["batch_size"],
                                     shuffle=False)

    solver = Solver(CONFIG,
                    train_loader1=train_loader1,
                    train_loader2=train_loader2,
                    valid_loader=val_loader,
                    tag_list=tag_list)
    solver.train()

# --------------------------- 主入口 ---------------------------
if __name__ == "__main__":
    # 在 CUDA 环境下安全启动多进程
    mp.set_start_method("spawn", force=True)
    train()

