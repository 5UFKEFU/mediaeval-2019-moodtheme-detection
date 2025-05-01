import os
import csv
import numpy as np
import torch
from torch.utils import data
import librosa
from transforms import get_transforms
import json

def get_spectrogram(full_spect, size=1400):
    full_len = full_spect.shape[1]
    if full_len > size:
        audio = full_spect[:, :size]
    else:
        diff = size - full_len
        audio = full_spect
        while diff > 0:
            if diff > full_len:
                audio = np.concatenate((audio, full_spect), axis=1)
                diff -= full_len
            else:
                audio = np.concatenate((audio, full_spect[:, :diff]), axis=1)
                diff = 0
    return audio

def normalize_tag(tag, tag_map):
    """使用tag_map.json规范化标签"""
    # 移除前缀
    if '---' in tag:
        tag_type, tag_value = tag.split('---', 1)
    else:
        return tag  # 如果不是标准格式，直接返回
        
    # 在对应的映射中查找
    if tag_type in tag_map and tag_value in tag_map[tag_type]:
        return f"{tag_type}---{tag_map[tag_type][tag_value]}"
    return tag

class AudioFolder(data.Dataset):
    def __init__(self, root, tsv_paths, labels_to_idx, num_classes=104, spect_len=4096, train=True):
        self.train = train
        self.root = root
        self.num_classes = num_classes
        self.spect_len = spect_len
        self.labels_to_idx = labels_to_idx
        
        self.prepare_data(tsv_paths)

        self.transform = get_transforms(
            train=train,
            size=spect_len,
            wrap_pad_prob=0.5 if train else 0.0,
            resize_scale=(0.8, 1.0) if train else (1.0, 1.0),
            resize_ratio=(1.7, 2.3) if train else (2.0, 2.0),
            resize_prob=0.33 if train else 0.0,
            spec_num_mask=2 if train else 0,
            spec_freq_masking=0.15 if train else 0.0,
            spec_time_masking=0.20 if train else 0.0,
            spec_prob=0.5 if train else 0.0
        )

    def __getitem__(self, index):
        # 读取 mp3 文件
        fn = os.path.join(self.root, self.paths[index])
        try:
            y, sr = librosa.load(fn, sr=16000, mono=True, duration=60)  # 采样率 16k，限制最多 60秒
        except Exception as e:
            print(f"Error loading file {fn}: {str(e)}")
            # Return a zero tensor with the same shape as expected
            y = np.zeros(16000 * 60)  # 60 seconds of silence at 16kHz
            sr = 16000
    
        # 提取 log-mel 特征
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel_spec = librosa.power_to_db(mel_spec)
    
        # 不要 expand_dims
        full_spect = log_mel_spec  # (freq, time)
    
        # 适配网络要求的大小
        audio = get_spectrogram(full_spect, self.spect_len)  # get_spectrogram 默认已经是 (freq, time)
        
        # 这里 self.transform 要求输入 2D，如果 transform 里面需要 Image.fromarray，要确保是 (H, W)，而不是 (1, H, W)
        audio = self.transform(audio)
    
        # 之后返回 audio + labels
        tags = self.tags[index]
        labels = self.one_hot(tags)
        return audio, labels
        
    
    def __len__(self):
        return len(self.paths)

    def one_hot(self, tags):
        labels = torch.LongTensor(tags)
        target = torch.zeros(self.num_classes).scatter_(0, labels, 1)
        return target

    def prepare_data(self, tsv_paths):
        all_dict = {
            'PATH': [],
            'TAGS': []
        }
        
        # 处理多个标签文件
        for tsv_path in tsv_paths:
            with open(tsv_path) as tsvfile:
                tsvreader = csv.reader(tsvfile, delimiter="\t")
                next(tsvreader)  # 跳过表头
                for line in tsvreader:
                    if len(line) < 6:  # 确保行有足够的列
                        continue
                        
                    if line[3] not in all_dict['PATH']:  # 如果路径不存在，添加新记录
                        all_dict['PATH'].append(line[3])
                        all_dict['TAGS'].append([])
                    
                    # 找到对应的路径索引
                    idx = all_dict['PATH'].index(line[3])
                    # 处理标签，从第6列开始
                    tag = line[5].strip()  # 获取标签并去除空白字符
                    if tag in self.labels_to_idx:
                        all_dict['TAGS'][idx].append(self.labels_to_idx[tag])

        self.paths = all_dict['PATH']
        self.tags = all_dict['TAGS']

def get_audio_loader(root, tsv_paths, labels_to_idx, batch_size=32, num_workers=8, shuffle=True, drop_last=True, train=True):
    dataset = AudioFolder(root, tsv_paths, labels_to_idx, num_classes=104, train=train)
    loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        prefetch_factor=2,
        persistent_workers=True
    )
    return loader