# split_autotagging_moodtheme_clean.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import csv

# 官方标准56个标签（需要根据官方文件完整列出）
official_labels = [
    "mood/theme---action", "mood/theme---adventure", "mood/theme---advertising",
    "mood/theme---background", "mood/theme---calm", "mood/theme---children",
    "mood/theme---christmas", "mood/theme---commercial", "mood/theme---cool",
    "mood/theme---corporate", "mood/theme---dark", "mood/theme---deep",
    "mood/theme---documentary", "mood/theme---dreamy", "mood/theme---driving",
    "mood/theme---emotional", "mood/theme---energetic", "mood/theme---epic",
    "mood/theme---fast", "mood/theme---film", "mood/theme---fun", "mood/theme---fusion",
    "mood/theme---game", "mood/theme---groovy", "mood/theme---happy", "mood/theme---heavy",
    "mood/theme---holiday", "mood/theme---hopeful", "mood/theme---house", "mood/theme---inspiring",
    "mood/theme---light", "mood/theme---love", "mood/theme---meditative", "mood/theme---melancholic",
    "mood/theme---motivational", "mood/theme---nature", "mood/theme---party", "mood/theme---positive",
    "mood/theme---powerful", "mood/theme---relaxing", "mood/theme---retro", "mood/theme---road-trip",
    "mood/theme---romantic", "mood/theme---sad", "mood/theme---sexy", "mood/theme---slow",
    "mood/theme---smooth", "mood/theme---soft", "mood/theme---space", "mood/theme---sport",
    "mood/theme---summer", "mood/theme---travel", "mood/theme---upbeat", "mood/theme---warm"
]

input_tsv = "../data/autotagging_moodtheme.tsv"
output_dir = "../data/"

# 读入大表
df = pd.read_csv(input_tsv, sep="\t", quoting=csv.QUOTE_NONE, engine='python', on_bad_lines='skip')

# 清洗非法标签
def clean_tags(tag_string):
    tags = tag_string.split(',')
    tags = [tag.strip() for tag in tags if tag.strip() in official_labels]
    return ",".join(tags)

df['TAGS'] = df['TAGS'].apply(clean_tags)

# 删除空标签行
df = df[df['TAGS'] != ""]

# 切分数据（80% 训练，10% 验证，10% 测试）
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# 保存
train_df.to_csv(os.path.join(output_dir, "autotagging_moodtheme-train.tsv"), sep="\t", index=False)
valid_df.to_csv(os.path.join(output_dir, "autotagging_moodtheme-valid.tsv"), sep="\t", index=False)
test_df.to_csv(os.path.join(output_dir, "autotagging_moodtheme-test.tsv"), sep="\t", index=False)

print("✅ 切分并清洗完成！train/valid/test 文件已生成。")
