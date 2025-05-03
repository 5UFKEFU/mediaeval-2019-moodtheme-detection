import os
import torch
import librosa
import numpy as np
import pandas as pd
from model import MusicSelfAttModel
from dataloader import get_spectrogram
from transforms import get_transforms
from collections import defaultdict

# 从all_tags.txt读取标签
def load_labels(label_file):
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels

# 加载标签
#LABEL_FILE = 'data/tags/all_tags.txt'
LABEL_FILE = 'data/tags/moodtheme_split.txt'

OFFICIAL_LABEL_LIST = load_labels(LABEL_FILE)
OFFICIAL_LABELS = {tag: tag.split('---')[1] for tag in OFFICIAL_LABEL_LIST}

def validate_model(model_path, tsv_file, num_samples=100):
    """验证模型在真实数据上的表现"""
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicSelfAttModel()
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # 读取TSV文件
    with open(tsv_file, 'r', encoding='utf-8') as f:
        # 读取标题行
        header = f.readline().strip().split('\t')
        # 读取数据行
        data = []
        for line in f:
            parts = line.strip().split('\t')
            # 确保前5个字段正确
            track_info = parts[:5]
            # 合并所有标签（第5个字段之后的所有内容）
            tags = '\t'.join(parts[5:])
            data.append(track_info + [tags])
    
    # 创建DataFrame
    df = pd.DataFrame(data, columns=header)
    
    # 随机采样
    sample_df = df.sample(n=min(num_samples, len(df)))
    
    # 设置音频预处理参数
    spect_len = 4096
    transform = get_transforms(
        train=False,
        size=spect_len,
        wrap_pad_prob=0.0,
        resize_scale=(1.0, 1.0),
        resize_ratio=(2.0, 2.0),
        resize_prob=0.0,
        spec_num_mask=0,
        spec_freq_masking=0.0,
        spec_time_masking=0.0,
        spec_prob=0.0
    )
    
    # 用于统计各类标签的准确率
    theme_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    genre_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    instrument_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    results = []
    for _, row in sample_df.iterrows():
        try:
            print("\n" + "="*80)  # 添加分隔线
            # 统一标签格式，确保使用 mood/theme
            ground_truth_tags = [tag if tag.startswith('mood/theme---') else f"mood/theme---{tag.split('---')[1]}" 
                               for tag in row['TAGS'].split('\t') if tag]
            
            # 加载音频文件
            audio_path = os.path.join('data/audio', row['PATH'])
            y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=60)
            
            # 提取特征
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            log_mel_spec = librosa.power_to_db(mel_spec)
            audio = get_spectrogram(log_mel_spec, spect_len)
            audio = transform(audio)
            audio = audio.unsqueeze(0).to(device)
            
            # 获取模型预测
            with torch.no_grad():
                att, clf = model(audio)
                all_scores = ((att[0] + clf[0]) / 2).tolist()
                
                # 获取每个类别的前几个预测
                theme_indices = [i for i, tag in enumerate(OFFICIAL_LABEL_LIST) if tag.startswith('mood/theme---')]
                genre_indices = [i for i, tag in enumerate(OFFICIAL_LABEL_LIST) if tag.startswith('genre---')]
                instrument_indices = [i for i, tag in enumerate(OFFICIAL_LABEL_LIST) if tag.startswith('instrument---')]
                
                # 打印主题标签的预测分数（紧凑格式）
                theme_scores = [(all_scores[idx], idx) for idx in theme_indices if idx < len(all_scores)]
                theme_scores.sort(reverse=True)
                
                print("主题标签预测 (前1个):", end=" ")
                scores_str = " | ".join([f"{OFFICIAL_LABEL_LIST[idx].split('---')[1]}: {score:.3f}" 
                                       for score, idx in theme_scores[:1]])
                print(scores_str)
                
                # 获取每个类别的前几个预测
                theme_top = theme_scores[:5]
                genre_scores = [(all_scores[idx], idx) for idx in genre_indices if idx < len(all_scores)]
                genre_top = sorted(genre_scores, reverse=True)[:5]
                instrument_scores = [(all_scores[idx], idx) for idx in instrument_indices if idx < len(all_scores)]
                instrument_top = sorted(instrument_scores, reverse=True)[:5]
                
                predicted_indices = [idx for _, idx in theme_top + genre_top + instrument_top]
                predicted_tags = [OFFICIAL_LABEL_LIST[idx] for idx in predicted_indices]
            
            # 计算准确率指标
            correct_predictions = set(predicted_tags) & set(ground_truth_tags)
            precision = len(correct_predictions) / len(predicted_tags) if predicted_tags else 0
            recall = len(correct_predictions) / len(ground_truth_tags) if ground_truth_tags else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # 更新标签统计
            for tag in ground_truth_tags:
                if tag.startswith('mood/theme---'):
                    theme_stats[tag]['total'] += 1
                    if tag in predicted_tags:
                        theme_stats[tag]['correct'] += 1
                    else:
                        print(f"未预测: {tag.split('---')[1]}")

            results.append({
                'track_id': row['TRACK_ID'],
                'ground_truth': ground_truth_tags,
                'predictions': predicted_tags,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            # 紧凑输出结果
            print(f"曲目ID: {row['TRACK_ID']} | 真实: {', '.join(t.split('---')[1] for t in ground_truth_tags)} | "
                  f"预测: {', '.join(t.split('---')[1] for t in predicted_tags)} | "
                  f"P: {precision:.2f} R: {recall:.2f} F1: {f1:.2f}")
            
            # 模型输出分数分布（紧凑格式）
            print(f"模型输出: Att[{att.min().item():.3f}, {att.max().item():.3f}] "
                  f"Clf[{clf.min().item():.3f}, {clf.max().item():.3f}]")
            
        except Exception as e:
            print(f"处理曲目 {row['TRACK_ID']} 时出错: {str(e)}")
    
    # 计算并打印总体指标
    if results:
        avg_precision = sum(r['precision'] for r in results) / len(results)
        avg_recall = sum(r['recall'] for r in results) / len(results)
        avg_f1 = sum(r['f1'] for r in results) / len(results)
        
        print("\n" + "="*80)
        print("总体验证结果:")
        print(f"平均精确率: {avg_precision:.2f} | 平均召回率: {avg_recall:.2f} | 平均F1分数: {avg_f1:.2f}")
        print(f"处理曲目数量: {len(results)}")
        
        # 打印各类标签的准确率（紧凑格式）
        print("\n主题标签准确率:")
        for tag, stats in theme_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{tag.split('---')[1]}: {accuracy:.2f} ({stats['correct']}/{stats['total']})", end=" | ")
        print()

if __name__ == "__main__":
    # 配置参数
    model_path = "models/best_model.pth"
    tsv_file = "data/autotagging_moodtheme.tsv"
    num_samples = 100  # 验证的曲目数量
    
    # 运行验证
    validate_model(model_path, tsv_file, num_samples) 