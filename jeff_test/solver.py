#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练 / 验证 / 推理逻辑。
"""
import os, time, datetime, pickle, csv
import numpy as np
import tqdm, swanlab
from sklearn import metrics

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import MusicSelfAttModel

class Solver:
    def __init__(self, config, train_loader1, train_loader2,
                 valid_loader, tag_list):
        # ---------------- 训练参数 ----------------
        self.n_epochs   = 120
        self.lr         = 1e-4
        self.log_step   = 100
        self.batch_size = config["batch_size"]
        self.tag_list   = tag_list
        self.num_class  = 56

        # ---------------- 硬件 ----------------
        self.is_cuda = torch.cuda.is_available()
        self.device  = torch.device("cuda" if self.is_cuda else "cpu")

        # ---------------- 路径 / logger ----------------
        self.model_save_path = config["log_dir"]
        os.makedirs(self.model_save_path, exist_ok=True)
        self.model_fn = os.path.join(self.model_save_path, "best_model.pth")
        self.writer   = SummaryWriter(self.model_save_path)
        swanlab.init(project="mediaeval2019-moodtheme", name="train-run")

        # ---------------- 数据 ----------------
        self.data_loader1 = train_loader1
        self.data_loader2 = train_loader2
        self.valid_loader = valid_loader

        # ---------------- 模型 ----------------
        self.build_model()

    # ---------------- 构建模型 & 优化器 ----------------
    def build_model(self):
        model = MusicSelfAttModel()
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.BCELoss()  # 根据任务自行调整

    # ---------------- 保存 / 加载 ----------------
    def save(self): torch.save(self.model.state_dict(), self.model_fn)
    def load(self, fn=None):
        fn = fn or self.model_fn
        self.model.load_state_dict(torch.load(fn, map_location=self.device))

    # ---------------- 训练循环 ----------------
    def train(self):
        best_val_auc = 0.0
        for epoch in range(1, self.n_epochs + 1):
            # ----------- 训练 -----------
            self.model.train()
            epoch_loss = 0.0
            t0 = time.time()
            for (x1, y1), (x2, y2) in zip(self.data_loader1, self.data_loader2):
                # mix-up 示例
                alpha = 1.0
                lam   = np.random.beta(alpha, alpha)
                x = lam * x1 + (1 - lam) * x2
                y = lam * y1 + (1 - lam) * y2

                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # ----------- 验证 -----------
            val_auc = self.evaluate()
            print(f"[{epoch:03d}/{self.n_epochs}] "
                  f"loss={epoch_loss/len(self.data_loader1):.4f} "
                  f"val_auc={val_auc:.4f} "
                  f"time={time.time()-t0:.1f}s")

            # ----------- 保存最好模型 -----------
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                self.save()

    # ---------------- 验证 ----------------
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        y_true, y_prob = [], []
        for x, y in self.valid_loader:
            x = x.to(self.device)
            out = self.model(x)
            y_true.append(y.numpy())
            y_prob.append(out.cpu().numpy())
        y_true = np.concatenate(y_true)
        y_prob = np.concatenate(y_prob)
        try:
            auc = metrics.roc_auc_score(y_true, y_prob, average="macro")
        except ValueError:
            auc = 0.0
        return auc

