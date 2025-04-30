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
        self.grad_accum_steps = 4  # 减少梯度累积步数
        self.warmup_steps = 1000  # 减少预热步数
        self.grad_norm_threshold = 0.5  # 降低梯度范数阈值
        self.loss_scale = 0.001  # 降低损失缩放因子
        self.grad_scale = 0.01  # 降低梯度缩放因子
        self.batch_size = 128  # 增加批处理大小

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 启用 CUDA 优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # 设置 CUDA 性能优化
            torch.cuda.set_per_process_memory_fraction(0.95)  # 允许使用95%的GPU内存
            torch.cuda.empty_cache()  # 清理GPU缓存
            
            # 设置CUDA流
            self.stream = torch.cuda.Stream()
            
            # 打印GPU信息
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f}MB")
            print(f"CUDA版本: {torch.version.cuda}")

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
        
        # 使用ReduceLROnPlateau调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode='max',
            factor=0.5,
            patience=self.lr_patience,
            min_lr=self.min_lr
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

        # 添加GPU监控
        def print_gpu_utilization():
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB allocated, "
                      f"{torch.cuda.memory_reserved() / 1024**2:.1f}MB reserved")
                # 打印GPU利用率
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    print(f"GPU利用率: {info.gpu}%, 内存带宽利用率: {info.memory}%")
                except:
                    pass

        for ep in range(1, self.epochs + 1):
            self.model.train()
            running = 0.0

            for step, (batch1, batch2) in enumerate(zip(self.dl1, self.dl2), start=1):
                current_loss = None  # 初始化current_loss变量
                try:
                    # 使用CUDA流进行异步处理
                    with torch.cuda.stream(self.stream):
                        # 将数据移到 GPU 并启用异步传输
                        lam = np.random.beta(1.,1., size=batch1[0].size(0)).astype("float32")
                        lam_x = torch.from_numpy(lam).view(-1,1,1,1).to(self.device, non_blocking=True)
                        lam_y = torch.from_numpy(lam).view(-1,1).to(self.device, non_blocking=True)

                        # 预处理输入数据
                        x1 = batch1[0].to(self.device, non_blocking=True)
                        x2 = batch2[0].to(self.device, non_blocking=True)
                        
                        # 检查输入形状并纠正
                        # 确保输入是 (B, 1, 96, 1400) 形状
                        if x1.dim() == 3:  # (B, 96, 1400)
                            x1 = x1.unsqueeze(1)  # 添加通道维度 -> (B, 1, 96, 1400)
                        if x2.dim() == 3:  # (B, 96, 1400)
                            x2 = x2.unsqueeze(1)  # 添加通道维度 -> (B, 1, 96, 1400)
                        
                        # 检查通道数是否正确
                        if x1.size(1) != 1:
                            print(f"警告: x1 通道数不正确: {x1.size(1)}，调整为 1")
                            x1 = x1[:, 0:1]  # 只保留第一个通道
                        if x2.size(1) != 1:
                            print(f"警告: x2 通道数不正确: {x2.size(1)}，调整为 1")
                            x2 = x2[:, 0:1]  # 只保留第一个通道
                        
                        # 检查并处理无效值
                        x1 = torch.nan_to_num(x1, nan=0.0, posinf=1.0, neginf=-1.0)
                        x2 = torch.nan_to_num(x2, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        # 计算混合输入
                        x = lam_x * x1 + (1-lam_x) * x2
                        
                        # 标准化输入
                        x = (x - x.mean()) / (x.std() + 1e-8)
                        
                        # 添加输入值范围限制
                        x = torch.clamp(x, -3.0, 3.0)
                        
                        # 打印最终输入形状用于调试
                        if step == 1:
                            print(f"Model input shape: {x.shape}")
                        
                        y1 = batch1[1].to(self.device, non_blocking=True)
                        y2 = batch2[1].to(self.device, non_blocking=True)
                        y = lam_y * y1 + (1-lam_y) * y2

                        # 使用混合精度训练
                        with torch.amp.autocast('cuda'):
                            att, clf = self.model(x)
                            if att.dim() == 1:
                                att = att.unsqueeze(1)
                            if clf.dim() == 1:
                                clf = clf.unsqueeze(1)
                            
                            att = att.expand(-1, 56)
                            current_loss = self.compute_loss(att, clf, y)
                            current_loss = current_loss / self.grad_accum_steps

                        # 反向传播
                        self.scaler.scale(current_loss).backward()
                        
                        # 梯度累积
                        if (step + 1) % self.grad_accum_steps == 0:
                            # 梯度裁剪
                            self.scaler.unscale_(self.opt)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)
                            
                            # 更新参数
                            self.scaler.step(self.opt)
                            self.scaler.update()
                            
                            # 重置梯度
                            self.opt.zero_grad(set_to_none=True)

                        # 清理不需要的张量
                        del x, y, att, clf, lam_x, lam_y, x1, x2, y1, y2
                        torch.cuda.empty_cache()

                        if current_loss is not None:
                            running += current_loss.item()
                            
                            # 每100步打印一次GPU使用情况
                            if step % 100 == 0:
                                print_gpu_utilization()
                                
                            # 每log_step步打印一次训练信息
                            if step % self.log_step == 0:
                                print(f"Epoch [{ep}/{self.epochs}] Step [{step}/{len(self.dl1)}] "
                                      f"Loss: {current_loss.item():.4f} "
                                      f"LR: {self.opt.param_groups[0]['lr']:.2e}")
                                
                                # 记录到tensorboard
                                self.writer.add_scalar('Loss/train', current_loss.item(), 
                                                     (ep-1)*len(self.dl1) + step)
                                self.writer.add_scalar('LR', self.opt.param_groups[0]['lr'], 
                                                     (ep-1)*len(self.dl1) + step)
                                
                                # 记录到swanlab
                                swanlab.log({
                                    "train_loss": current_loss.item(),
                                    "learning_rate": self.opt.param_groups[0]['lr']
                                })
                                
                                # 检查梯度
                                grad_norm = self.check_gradients()
                                if grad_norm > self.grad_norm_threshold:
                                    print(f"Warning: Gradient norm ({grad_norm:.4f}) exceeds threshold")
                                    self.scale_gradients()
                                    
                                # 检查NaN
                                if torch.isnan(current_loss):
                                    nan_count += 1
                                    print(f"Warning: NaN loss detected ({nan_count} times)")
                                    if nan_count >= 3:
                                        print("Too many NaN losses, stopping training")
                                        return
                                        
                                # 检查学习率
                                if self.opt.param_groups[0]['lr'] < self.min_lr:
                                    lr_patience_count += 1
                                    if lr_patience_count >= self.lr_patience:
                                        print("Learning rate too small for too long, stopping training")
                                        return
                                        
                                # 累积损失
                                accumulated_loss += current_loss.item()
                                
                                # 每1000步评估一次
                                if step % 1000 == 0:
                                    val_auc = self.evaluate()
                                    self.scheduler.step(val_auc)  # 使用验证集AUC来调整学习率

                except Exception as e:
                    print(f"Error in step {step}: {str(e)}")
                    # 清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

            # 等待所有CUDA操作完成
            torch.cuda.synchronize()

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
            # 确保输入是 (B, 1, 96, 1400) 形状
            if x.dim() == 3:  # (B, 96, 1400)
                x = x.unsqueeze(1)  # 添加通道维度 -> (B, 1, 96, 1400)
            
            # 检查通道数是否正确
            if x.size(1) != 1:
                print(f"警告: 验证数据通道数不正确: {x.size(1)}，调整为 1")
                x = x[:, 0:1]  # 只保留第一个通道
            
            # 检查并处理无效值
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 标准化输入
            x = (x - x.mean()) / (x.std() + 1e-8)
            
            # 添加输入值范围限制
            x = torch.clamp(x, -3.0, 3.0)
            
            # 将数据移到设备上
            x = x.to(self.device)
            
            # 使用模型进行预测
            att, clf = self.model(x)
            ps.append(clf.cpu())
            ys.append(y)
        ys, ps = torch.cat(ys).numpy(), torch.cat(ps).numpy()
        try:
            return metrics.roc_auc_score(ys, ps, average="macro")
        except ValueError:
            return 0.0

    @torch.no_grad()
    def test(self):
        self.model.eval()
        outputs = []
        
        for batch in self.dl1:
            try:
                x = batch[0].to(self.device)
                
                # 检查输入形状并纠正
                # 确保输入是 (B, 1, 96, 1400) 形状
                if x.dim() == 3:  # (B, 96, 1400)
                    x = x.unsqueeze(1)  # 添加通道维度 -> (B, 1, 96, 1400)
                
                # 检查通道数是否正确
                if x.size(1) != 1:
                    print(f"警告: 测试数据通道数不正确: {x.size(1)}，调整为 1")
                    x = x[:, 0:1]  # 只保留第一个通道
                
                # 检查并处理无效值
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 标准化输入
                x = (x - x.mean()) / (x.std() + 1e-8)
                
                # 添加输入值范围限制
                x = torch.clamp(x, -3.0, 3.0)
                
                with torch.amp.autocast('cuda'):
                    att, clf = self.model(x)
                    o = torch.sigmoid(0.5 * (att + clf))
                    outputs.append(o.cpu().numpy())
            except Exception as e:
                print(f"测试时出错: {str(e)}")
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
                
        return np.vstack(outputs)