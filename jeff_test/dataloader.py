#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset & DataLoader。全部在 CPU 上做预处理，最后在 Solver 中一次性迁移到 GPU。
"""
import os, csv, torch, torchaudio
import numpy as np
from torch.utils import data
import torchaudio.transforms as T

# ----------- 公共工具 -----------
def get_spectrogram(full_spec, size=1400):
    full_len = full_spec.shape[1]
    if full_len > size:
        audio = full_spec[:, :size]
    else:
        diff = size - full_len
        audio = full_spec
        while diff > 0:
            if diff > full_len:
                audio = np.concatenate((audio, full_spec), axis=1)
                diff -= full_len
            else:
                audio = np.concatenate((audio, full_spec[:, :diff]), axis=1)
                diff = 0
    return audio

class AudioFolder(data.Dataset):
    def __init__(self, root, tsv_path, labels_to_idx,
                 num_classes=56, spect_len=4096, train=True):
        self.root, self.train = root, train
        self.num_classes = num_classes
        self.spect_len   = spect_len

        # 读取 tsv
        self.paths, self.labels = [], []
        with open(tsv_path) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                self.paths.append(row[0])
                # one-hot
                tgt = np.zeros(num_classes, dtype=np.float32)
                for lab in row[1].split(","):
                    if lab in labels_to_idx:
                        tgt[labels_to_idx[lab]] = 1.0
                self.labels.append(tgt)

        # MelSpectrogram 在 CPU 构建
        self.mel = T.MelSpectrogram(
            sample_rate=22050, n_fft=1024, hop_length=512,
            n_mels=96
        )

    def __len__(self): return len(self.paths)

    def __getitem__(self, index):
        mp3_path = os.path.join(self.root, self.paths[index])
        waveform, sr = torchaudio.load(mp3_path)

        # 重采样 → 22050 Hz
        if sr != 22050:
            resampler = T.Resample(sr, 22050)
            waveform  = resampler(waveform)

        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        spec = self.mel(waveform)          # (1, n_mels, T)
        spec = torch.log(spec + 1e-9)      # 对数振幅
        spec = get_spectrogram(spec.squeeze(0).numpy())  # -> (96, 1400)
        spec = torch.from_numpy(spec).float()

        label = torch.from_numpy(self.labels[index])
        return spec, label

# --------- DataLoader 帮助函数 ----------
def get_audio_loader(root, tsv_path, labels_to_idx, batch_size=8, shuffle=True):
    dataset = AudioFolder(root, tsv_path, labels_to_idx)
    loader  = data.DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=4,
                              pin_memory=True)
    return loader

