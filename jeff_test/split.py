# split_autotagging_moodtheme_clean.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import csv
import json

# 加载标签映射
with open("data/tag_map.json", 'r', encoding='utf-8') as f:
    TAG_MAP = json.load(f)

# 创建官方标签列表
official_labels = []

# 添加mood/theme标签
for tag in sorted(set(TAG_MAP['theme'].values())):
    official_labels.append(f"theme---{tag}")

# 添加genre标签
for tag in sorted(set(TAG_MAP['genre'].values())):
    official_labels.append(f"genre---{tag}")

# 添加instrument标签
for tag in sorted(set(TAG_MAP['instrument'].values())):
    official_labels.append(f"instrument---{tag}")

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

def clean_tags(tag_string, tag_map):
    """清洗和规范化标签"""
    tags = tag_string.split(',')
    normalized_tags = []
    for tag in tags:
        tag = tag.strip()
        if not tag:
            continue
            
        # 规范化标签
        normalized_tag = normalize_tag(tag, tag_map)
        if normalized_tag in official_labels:
            normalized_tags.append(normalized_tag)
    
    return ",".join(normalized_tags)

def process_tsv_file(input_tsv, output_dir, tag_map):
    """处理单个TSV文件"""
    # 读入大表
    df = pd.read_csv(input_tsv, sep="\t", quoting=csv.QUOTE_NONE, engine='python', on_bad_lines='skip')

    # 清洗非法标签
    df['TAGS'] = df['TAGS'].apply(lambda x: clean_tags(x, tag_map))

    # 删除空标签行
    df = df[df['TAGS'] != ""]

    # 切分数据（80% 训练，10% 验证，10% 测试）
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # 获取文件名前缀
    base_name = os.path.basename(input_tsv).split('.')[0]

    # 获取类型名（去掉前缀和后缀）
    type_name = base_name.replace('autotagging_', '')

    # 保存
    train_df.to_csv(os.path.join(output_dir, f"{type_name}-train.tsv"), sep="\t", index=False)
    valid_df.to_csv(os.path.join(output_dir, f"{type_name}-valid.tsv"), sep="\t", index=False)
    test_df.to_csv(os.path.join(output_dir, f"{type_name}-test.tsv"), sep="\t", index=False)

    return len(train_df), len(valid_df), len(test_df)

def main():
    input_dir = "data"
    output_dir = "data"
    
    # 处理所有标签文件
    tsv_files = [
        "autotagging_moodtheme.tsv",
        "autotagging_genre.tsv",
        "autotagging_instrument.tsv"
    ]
    
    total_train = 0
    total_valid = 0
    total_test = 0
    
    for tsv_file in tsv_files:
        input_tsv = os.path.join(input_dir, tsv_file)
        if not os.path.exists(input_tsv):
            print(f"警告：文件 {input_tsv} 不存在，跳过")
            continue
            
        print(f"\n处理文件: {tsv_file}")
        train_size, valid_size, test_size = process_tsv_file(input_tsv, output_dir, TAG_MAP)
        
        total_train += train_size
        total_valid += valid_size
        test_size += test_size
        
        print(f"训练集大小: {train_size}")
        print(f"验证集大小: {valid_size}")
        print(f"测试集大小: {test_size}")
    
    print("\n✅ 所有文件切分并清洗完成！")
    print(f"总训练集大小: {total_train}")
    print(f"总验证集大小: {total_valid}")
    print(f"总测试集大小: {total_test}")

if __name__ == "__main__":
    main()
