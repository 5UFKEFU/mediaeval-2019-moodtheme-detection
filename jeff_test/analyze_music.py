import os
import torch
import librosa
import numpy as np
from model import MusicSelfAttModel
from dataloader import get_spectrogram
from torch.utils.data import DataLoader
import requests
from bs4 import BeautifulSoup
import re
import json
from tqdm import tqdm
from transforms import get_transforms
import time
import hashlib
import base64
import binascii
import random
import string
import openai
from collections import defaultdict

# 设置 OpenAI API 密钥
openai.api_key = os.getenv('OPENAI_API_KEY')

# 加载标签映射
with open('../data/data/tag_map.json', 'r', encoding='utf-8') as f:
    TAG_MAP = json.load(f)

# 创建官方标签列表
OFFICIAL_LABELS = {}
OFFICIAL_LABEL_LIST = []

# 添加mood/theme标签
for tag in sorted(set(TAG_MAP['mood/theme'].values())):
    OFFICIAL_LABELS[f"mood/theme---{tag}"] = tag
    OFFICIAL_LABEL_LIST.append(f"mood/theme---{tag}")

# 添加genre标签
for tag in sorted(set(TAG_MAP['genre'].values())):
    OFFICIAL_LABELS[f"genre---{tag}"] = tag
    OFFICIAL_LABEL_LIST.append(f"genre---{tag}")

# 添加instrument标签
for tag in sorted(set(TAG_MAP['instrument'].values())):
    OFFICIAL_LABELS[f"instrument---{tag}"] = tag
    OFFICIAL_LABEL_LIST.append(f"instrument---{tag}")

# 用于记录非官方标签的字典
non_official_tags = defaultdict(int)

def extract_song_info(filename):
    """从文件名中提取歌曲名和演唱者，只取第一个'-'分隔"""
    if filename.lower().endswith('.mp3'):
        base = filename[:-4]
        parts = base.split('-', 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    return None, None

# Function to get tags using a large language model

def get_llm_tags(song_name, artist):
    """Use a large language model to search for online perception tags"""
    try:
        print(f"\n使用 LLM 分析歌曲 {song_name} - {artist}...")
        # 构建查询
        query = f"""请分析歌曲 '{song_name}' 的演唱者 '{artist}' 的风格、情感特征和使用的乐器。
请提供以下类型的标签（用逗号分隔）：
1. 情感/主题标签（最多3个）
2. 乐器标签（最多3个）
3. 风格标签（最多2个）

如果标签不在以下列表中，也请提供，但请标注为"非官方标签"：
{', '.join(OFFICIAL_LABEL_LIST)}"""
        
        # 调用 LLM API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个音乐分析专家，擅长分析歌曲的风格、情感特征和使用的乐器。"},
                {"role": "user", "content": query}
            ],
            max_tokens=200
        )
        
        # 提取标签
        response_text = response.choices[0].message.content
        tags = [tag.strip() for tag in response_text.split(',')]
        
        # 分离官方标签和非官方标签
        valid_tags = []
        for tag in tags:
            if tag in OFFICIAL_LABEL_LIST:
                valid_tags.append(tag)
            else:
                # 记录非官方标签
                non_official_tags[tag] += 1
                print(f"发现非官方标签: {tag}")
        
        print(f"从 LLM 获取的标签: {valid_tags}")
        return valid_tags
    except Exception as e:
        print(f"从 LLM 获取标签时出错: {e}")
        return []

# Modify the get_online_tags function to use the LLM

def get_online_tags(song_name, artist):
    """Get song tags from multiple sources including LLM"""
    # Clean song name and artist name
    song_name = song_name.split('-')[0].strip()  # Remove extra info from filename
    artist = artist.split('-')[0].strip()
    
    print(f"\n正在获取 {song_name} - {artist} 的在线标签...")
    
    # Get tags from LLM
    llm_tags = get_llm_tags(song_name, artist)
    
    # Combine and deduplicate tags
    all_tags = list(set(llm_tags))
    print(f"最终合并的标签: {all_tags}")
    
    return all_tags

def analyze_music_files(directory, model_path):
    """分析目录中的音乐文件"""
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicSelfAttModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
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
    
    # 获取所有MP3文件
    mp3_files = [f for f in os.listdir(directory) if f.endswith('.mp3')]
    
    results = []
    for filename in tqdm(mp3_files):
        song_name, artist = extract_song_info(filename)
        if not song_name or not artist:
            print(f"无法解析文件名: {filename}")
            continue
            
        # 加载音频文件
        audio_path = os.path.join(directory, filename)
        try:
            # 使用与训练时相同的预处理
            y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=60)
            
            # 提取 log-mel 特征
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            log_mel_spec = librosa.power_to_db(mel_spec)
            
            # 适配网络要求的大小
            audio = get_spectrogram(log_mel_spec, spect_len)
            
            # 应用转换
            audio = transform(audio)
            
            # 添加batch维度
            audio = audio.unsqueeze(0).to(device)
            
            # 模型预测
            with torch.no_grad():
                att, clf = model(audio)
                predicted_indices = clf.topk(5)[1].tolist()[0]  # 获取前5个预测标签的索引
                predicted_tags = [OFFICIAL_LABEL_LIST[idx] for idx in predicted_indices]  # 转换为实际标签
            
            # 获取在线标签
            online_tags = get_online_tags(song_name, artist)
            
            results.append({
                'filename': filename,
                'song_name': song_name,
                'artist': artist,
                'model_predictions': predicted_tags,
                'model_predictions_indices': predicted_indices,
                'online_tags': online_tags
            })
            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
    
    return results

def save_results(results, output_file):
    """保存分析结果到JSON文件"""
    # 添加中文标签到结果中
    for result in results:
        result['model_predictions_chinese'] = [OFFICIAL_LABELS[tag] for tag in result['model_predictions']]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def save_non_official_tags(output_file="non_official_tags.json"):
    """保存非官方标签统计到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dict(non_official_tags), f, ensure_ascii=False, indent=2)
    print(f"\n非官方标签统计已保存到 {output_file}")
    print(f"共发现 {len(non_official_tags)} 个不同的非官方标签")
    print("标签出现次数统计：")
    for tag, count in sorted(non_official_tags.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tag}: {count} 次")

if __name__ == "__main__":
    # 配置参数
    music_directory = "test_music/"  # 替换为实际的音乐文件目录
    model_path = "models/best_model.pth"  # 替换为实际的模型路径
    output_file = "analysis_results.json"
    
    # 分析音乐文件
    results = analyze_music_files(music_directory, model_path)
    
    # 保存结果
    save_results(results, output_file)
    
    # 保存非官方标签统计
    save_non_official_tags()
    
    print(f"分析完成！结果已保存到 {output_file}")
    
    # 打印结果摘要
    print("\n分析结果摘要：")
    for result in results:
        print(f"\n歌曲: {result['song_name']} - {result['artist']}")
        print("模型预测标签:")
        for tag in result['model_predictions']:
            print(f"  - {tag} ({OFFICIAL_LABELS[tag]})")
        print("在线标签:")
        for tag in result['online_tags']:
            print(f"  - {tag} ({OFFICIAL_LABELS[tag]})") 

# Ensure to add your OpenAI API key
#openai.api_key_path = 'models/openai_api_key.txt' 