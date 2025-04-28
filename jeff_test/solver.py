#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练 / 验证 —— minimal GPU 适配
"""
import os, time, datetime, numpy as np, torch, torch.nn as nn
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
import swanlab

from model import MusicSelfAttModel

class Solver:
    def __init__(self, data_loader1, data_loader2,
                 valid_loader, tag_list, cfg):
        # 数据
        self.dl1, self.dl2, self.val = data_loader1, data_loader2, valid_loader

        # 训练超参
        self.epochs = 120
        self.lr     = 1e-7  # 进一步降低学习率
        self.log_step = 1
        self.max_grad_norm = 0.05  # 降低梯度裁剪阈值
        self.grad_clip_val = 0.005  # 降低梯度裁剪值
        self.nan_threshold = 1e-6
        self.min_lr = 1e-7
        self.lr_patience = 3
        self.grad_accum_steps = 64  # 增加梯度累积步数
        self.warmup_steps = 2000  # 增加预热步数
        self.grad_norm_threshold = 0.5  # 降低梯度范数阈值
        self.loss_scale = 0.001  # 降低损失缩放因子
        self.grad_scale = 0.01  # 降低梯度缩放因子

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 启用 CUDA 优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # 日志
        os.makedirs(cfg["log_dir"], exist_ok=True)
        self.writer = SummaryWriter(cfg["log_dir"])

        swanlab.init(project="mediaeval2019-moodtheme", name="train-run")

        # 模型
        self.model = MusicSelfAttModel().to(self.device)
        
        # 改进的权重初始化
        for m in self.model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 使用AdamW优化器，添加权重衰减
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.1,  # 增加权重衰减
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 添加学习率调度器
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.opt,
            max_lr=self.lr,
            epochs=self.epochs,
            steps_per_epoch=len(self.dl1),
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        self.crit = nn.BCEWithLogitsLoss(reduction='none')
        
        # 初始化混合精度训练
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    def check_gradients(self):
        """检查梯度是否在合理范围内"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def scale_gradients(self):
        """缩放梯度"""
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(self.grad_scale)

    def compute_loss(self, att, clf, y):
        """计算损失，添加数值稳定性检查"""
        # 计算注意力损失
        att_loss = self.crit(att, y)
        att_loss = torch.mean(att_loss)
        
        # 计算分类损失
        clf_loss = self.crit(clf, y)
        clf_loss = torch.mean(clf_loss)
        
        # 检查损失值
        if torch.isnan(att_loss) or torch.isinf(att_loss):
            att_loss = torch.tensor(0.0, device=self.device)
        if torch.isnan(clf_loss) or torch.isinf(clf_loss):
            clf_loss = torch.tensor(0.0, device=self.device)
            
        # 组合损失并应用缩放
        total_loss = 0.5 * (att_loss + clf_loss) * self.loss_scale
        return total_loss

    # ---------- 训练 ----------
    def train(self):
        print("Training on", self.device)
        best_auc = 0.0
        start = time.time()
        nan_count = 0
        lr_patience_count = 0
        accumulated_loss = 0.0

        for ep in range(1, self.epochs + 1):
            self.model.train()
            running = 0.0

            for step, (batch1, batch2) in enumerate(zip(self.dl1, self.dl2), start=1):
                # 清理缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 将数据移到 GPU 并启用异步传输
                lam = np.random.beta(1.,1., size=batch1[0].size(0)).astype("float32")
                lam_x = torch.from_numpy(lam).view(-1,1,1,1).to(self.device, non_blocking=True)
                lam_y = torch.from_numpy(lam).view(-1,1).to(self.device, non_blocking=True)

                # 预处理输入数据
                x1 = batch1[0].to(self.device, non_blocking=True)
                x2 = batch2[0].to(self.device, non_blocking=True)
                
                # 检查并处理无效值
                x1 = torch.nan_to_num(x1, nan=0.0, posinf=1.0, neginf=-1.0)
                x2 = torch.nan_to_num(x2, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 计算混合输入
                x = lam_x * x1 + (1-lam_x) * x2
                
                # 标准化输入
                x = (x - x.mean()) / (x.std() + 1e-8)
                
                # 添加输入值范围限制
                x = torch.clamp(x, -3.0, 3.0)  # 更严格的范围限制
                
                y1 = batch1[1].to(self.device, non_blocking=True)
                y2 = batch2[1].to(self.device, non_blocking=True)
                y = lam_y * y1 + (1-lam_y) * y2

                # 检查输入数据
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"Warning: NaN/Inf in input data at step {step}")
                    continue

                # 使用新的混合精度训练API
                with torch.amp.autocast('cuda'):
                    att, clf = self.model(x)
                    if step % 50 == 0:
                        print(f"att shape: {att.shape}, clf shape: {clf.shape}, y shape: {y.shape}")
                    
                    if att.dim() == 1:
                        att = att.unsqueeze(1)
                    if clf.dim() == 1:
                        clf = clf.unsqueeze(1)
                    
                    att = att.expand(-1, 56)
                    loss = self.compute_loss(att, clf, y)
                    loss = loss / self.grad_accum_steps  # 缩放损失

                # 检查loss是否为nan
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    print(f"Warning: NaN/Inf loss detected at step {step}, count: {nan_count}")
                    if nan_count > 5:
                        lr_patience_count += 1
                        if lr_patience_count >= self.lr_patience:
                            print("Too many NaN losses, reducing learning rate...")
                            for param_group in self.opt.param_groups:
                                new_lr = param_group['lr'] * 0.5
                                if new_lr >= self.min_lr:
                                    param_group['lr'] = new_lr
                                    print(f"Learning rate reduced to {new_lr}")
                                else:
                                    print("Learning rate already at minimum")
                            lr_patience_count = 0
                        nan_count = 0
                    continue

                # 保存loss值
                loss_value = loss.item() * self.grad_accum_steps  # 恢复原始scale
                accumulated_loss += loss_value

                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 检查梯度是否为nan
                has_nan_grad = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"Warning: NaN/Inf gradient in {name}")
                            param.grad.data.zero_()
                            has_nan_grad = True
                
                if has_nan_grad:
                    print("Skipping batch due to NaN gradients")
                    self.opt.zero_grad(set_to_none=True)
                    continue

                # 梯度累积
                if (step + 1) % self.grad_accum_steps == 0:
                    # 检查梯度范数
                    grad_norm = self.check_gradients()
                    if grad_norm > self.grad_norm_threshold:
                        print(f"Warning: Gradient norm {grad_norm:.2f} exceeds threshold {self.grad_norm_threshold}")
                        self.opt.zero_grad(set_to_none=True)
                        continue
                    
                    # 缩放梯度
                    self.scale_gradients()
                    
                    # 梯度裁剪
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)
                    
                    # 更新参数
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    
                    # 更新学习率
                    self.scheduler.step()
                    
                    # 重置梯度
                    self.opt.zero_grad(set_to_none=True)

                # 清理不需要的张量
                del x, y, att, clf, loss, lam_x, lam_y, x1, x2, y1, y2
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                running += loss_value

                if step % self.log_step == 0:
                    elapsed = str(datetime.timedelta(seconds=int(time.time()-start)))
                    print(f"[{ep:03d}] iter {step:4d}/{len(self.dl1)} "
                          f"loss {running/self.log_step:.4f}  {elapsed}")
                    self.writer.add_scalar("train/loss_iter", running/self.log_step,
                                           (ep-1)*len(self.dl1)+step)
                    swanlab.log({"train/loss_iter": running/self.log_step})
                    running = 0.0

            # ---------- 验证 ----------
            auc = self.evaluate()
            print(f"Epoch {ep:03d}  AUC {auc:.4f}")
            swanlab.log({"val/auc": auc})
            self.writer.add_scalar("val/auc", auc, ep)

            # 存最好模型
            if auc > best_auc:
                best_auc = auc
                torch.save(self.model.state_dict(), "best_model.pth")

    # ---------- 验证 ----------
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        ys, ps = [], []
        for x, y in self.val:
            ps.append(self.model(x.to(self.device))[1].cpu())
            ys.append(y)
        ys, ps = torch.cat(ys).numpy(), torch.cat(ps).numpy()
        try:
            return metrics.roc_auc_score(ys, ps, average="macro")
        except ValueError:
            return 0.0