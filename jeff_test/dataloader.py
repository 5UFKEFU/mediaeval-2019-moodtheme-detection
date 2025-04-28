#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 mp3 直接生成 Mel-Spectrogram：
    • 解码 / 重采样 —— CPU
    • Mel + log          —— GPU (若可用)
返回：audio  (1, 96, 1400)  float32
        label  (56,)      float32
"""
import os, csv, torch, torchaudio, numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from transforms import get_transforms
import torch.nn.functional as F

TARGET_LEN = 1400                    # time 轴固定长度
N_MELS     = 96

# GPU 上预生成 Mel（FFT 非常快）
MEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEL = T.MelSpectrogram(
    sample_rate=22050,
    n_fft=1024,
    hop_length=512,
    n_mels=N_MELS
).to(MEL_DEVICE)

def pad_or_repeat(mel: np.ndarray, length: int = TARGET_LEN) -> np.ndarray:
    if mel.shape[1] >= length:
        return mel[:, :length]
    rep = length // mel.shape[1] + 1
    return np.tile(mel, (1, rep))[:, :length]

class AudioFolder(Dataset):
    def __init__(self, root, tsv_path, lab2idx,
                 num_cls: int = 56, train: bool = True):
        self.root = root
        self.num_cls = num_cls

        # 预加载所有标签到 CPU
        self.paths, self.labels = [], []
        with open(tsv_path) as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)                # 跳 header
            for row in reader:
                self.paths.append(row[3])                    # PATH
                tag_str = row[5].replace('---', ',')
                y = np.zeros(num_cls, np.float32)
                for t in tag_str.split(','):
                    t = t.strip()
                    if t and t in lab2idx:
                        y[lab2idx[t]] = 1.
                self.labels.append(y)
        
        # 将标签转换为 CPU tensor
        self.labels = torch.from_numpy(np.array(self.labels))

        self.transform = get_transforms(
            train=train,
            size=TARGET_LEN,
            wrap_pad_prob=0.5,
            resize_scale=(0.8,1.0),
            resize_ratio=(1.7,2.3),
            resize_prob=0.33,
            spec_num_mask=2,
            spec_freq_masking=0.15,
            spec_time_masking=0.20,
            spec_prob=0.5
        ) if train else get_transforms(False, TARGET_LEN)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        fp = os.path.join(self.root, self.paths[idx])
        wav, sr = torchaudio.load(fp)               # 解码 (CPU)

        if sr != 22050:
            wav = T.Resample(sr, 22050)(wav)
        wav = wav.mean(0, keepdim=True)

        # 将数据移到与MEL相同的设备上
        wav = wav.to(MEL_DEVICE)
        
        # 生成 mel spectrogram
        mel = MEL(wav).log()
        mel = mel.squeeze(0)
        
        # 在 GPU 上进行 padding
        if mel.shape[1] >= TARGET_LEN:
            mel = mel[:, :TARGET_LEN]
        else:
            rep = TARGET_LEN // mel.shape[1] + 1
            mel = torch.tile(mel, (1, rep))[:, :TARGET_LEN]
        
        # 应用数据增强
        mel = self.transform(mel)
        
        # 确保输出大小一致
        if mel.shape[0] != N_MELS:
            mel = mel.unsqueeze(0)
            mel = F.interpolate(
                mel,
                size=(N_MELS, TARGET_LEN),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # 将数据移回 CPU
        mel = mel.cpu()
        
        return mel, self.labels[idx]

def get_audio_loader(root, tsv, lab2idx,
                     batch_size=16, num_workers=1,  # 减少到1个worker
                     shuffle=True, drop_last=True):
    ds = AudioFolder(root, tsv, lab2idx, train=shuffle)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True,
                      drop_last=drop_last,
                      persistent_workers=True,
                      prefetch_factor=1,  # 减少预加载批次
                      multiprocessing_context='spawn')