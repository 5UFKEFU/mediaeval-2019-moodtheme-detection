import os
import torch
import librosa
import numpy as np
import pandas as pd
from model import MusicSelfAttModel
from dataloader import get_spectrogram
from transforms import get_transforms
from collections import defaultdict
import argparse

# 从all_tags.txt读取标签
def load_labels(label_file):
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels

# 加载标签
LABEL_FILE = 'data/tags/all_tags.txt'
#LABEL_FILE = 'data/tags/moodtheme_split.txt'

OFFICIAL_LABEL_LIST = load_labels(LABEL_FILE)
OFFICIAL_LABELS = {tag: tag.split('---')[1] for tag in OFFICIAL_LABEL_LIST}

def validate_model(model_path, tsv_file, num_samples=100, label_type='theme'):
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
    stats_dict = {
        'theme': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'genre': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'instrument': defaultdict(lambda: {'correct': 0, 'total': 0}),
    }
    
    results = []
    for _, row in sample_df.iterrows():
        try:
            print("\n" + "="*80)  # 添加分隔线
            # 统一标签格式
            ground_truth_tags = [tag if tag.startswith('mood/theme---') or tag.startswith('genre---') or tag.startswith('instrument---')
                                 else f"mood/theme---{tag.split('---')[1]}"
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
                
                # 根据label_type选择
                if label_type == 'theme':
                    indices = theme_indices
                    label_prefix = 'mood/theme---'
                elif label_type == 'genre':
                    indices = genre_indices
                    label_prefix = 'genre---'
                elif label_type == 'instrument':
                    indices = instrument_indices
                    label_prefix = 'instrument---'
                else:
                    raise ValueError(f"未知的label_type: {label_type}")
                
                scores = [(all_scores[idx], idx) for idx in indices if idx < len(all_scores)]
                scores.sort(reverse=True)
                # 只保留分数大于0.1的标签
                filtered_scores = [(score, idx) for score, idx in scores if score > 0.1]
                top_scores = filtered_scores[:5]  # 最多取前5个
                predicted_indices = [idx for score, idx in top_scores]
                predicted_tags = [OFFICIAL_LABEL_LIST[idx] for idx in predicted_indices]
                predicted_scores = [score for score, idx in top_scores]
            
            # 计算准确率指标
            gt_tags = [tag for tag in ground_truth_tags if tag.startswith(label_prefix)]
            correct_predictions = set(predicted_tags) & set(gt_tags)
            precision = len(correct_predictions) / len(predicted_tags) if predicted_tags else 0
            recall = len(correct_predictions) / len(gt_tags) if gt_tags else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # 更新标签统计
            for tag in gt_tags:
                stats_dict[label_type][tag]['total'] += 1
                if tag in predicted_tags:
                    stats_dict[label_type][tag]['correct'] += 1
                else:
                    pass

            results.append({
                'track_id': row['TRACK_ID'],
                'ground_truth': ground_truth_tags,
                'predictions': predicted_tags,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            # 紧凑输出结果（改为多行+分数+中/未中）
            print(f"曲目ID: {row['TRACK_ID']}")

            # 真实标签部分，标注"中"或"未中"
            gt_tag_names = []
            for tag in gt_tags:
                tag_name = tag.split('---')[1]
                if tag in predicted_tags:
                    gt_tag_names.append(f"{tag_name}(中)")
                else:
                    gt_tag_names.append(f"{tag_name}(未中)")
            print(f"真实: {', '.join(gt_tag_names)}")

            # 预测标签部分，带分数
            pred_strs = []
            for tag, score in zip(predicted_tags, predicted_scores):
                tag_name = tag.split('---')[1]
                pred_strs.append(f"{tag_name}({score:.1f})")
            print(f"预测: {', '.join(pred_strs)}")

            print(f"P: {precision:.2f} R: {recall:.2f} F1: {f1:.2f}")
            print(f"模型输出: Att[{att.min().item():.3f}, {att.max().item():.3f}] Clf[{clf.min().item():.3f}, {clf.max().item():.3f}]")
            
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
        print(f"\n{label_type}标签准确率:")
        for tag, stats in stats_dict[label_type].items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{tag.split('---')[1]}: {accuracy:.2f} ({stats['correct']}/{stats['total']})", end=" | ")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="output2/best_model.pth")
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--label_type', type=str, choices=['theme', 'genre', 'instrument'], default='theme')
    args = parser.parse_args()

    # 根据 label_type 选择 tsv 文件
    if args.label_type == 'theme':
        tsv_file = "data/autotagging_moodtheme.tsv"
    elif args.label_type == 'genre':
        tsv_file = "data/autotagging_genre.tsv"
    elif args.label_type == 'instrument':
        tsv_file = "data/autotagging_instrument.tsv"
    else:
        raise ValueError(f"未知的label_type: {args.label_type}")

    validate_model(args.model_path, tsv_file, args.num_samples, args.label_type) 