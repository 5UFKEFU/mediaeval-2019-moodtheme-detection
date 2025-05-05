import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
from pydub import AudioSegment
import os

# ==== 配置 ====
input_path = os.path.join(os.path.dirname(__file__), "刘德华-忘情水.mp3")
top_k = 10


# ==== 标签分类映射（可根据需要扩展） ====
CATEGORY_MAP = {
    "Genre": [
        "pop music", "rock music", "hip hop music", "jazz", "classical music", "electronic music",
        "reggae", "soul music", "funk", "blues", "country music", "techno", "metal"
    ],
    "Mood": [
        "happy music", "sad music", "angry music", "exciting music", "calm music",
        "romantic music", "scary music", "dark music", "serene music", "tense music"
    ],
    "Instrument": [
        "guitar", "drum", "piano", "violin", "saxophone", "trumpet", "flute", "cello",
        "harmonica", "synthesizer", "keyboard", "bass guitar", "electric guitar"
    ],
    "Scene": [
        "party", "wedding", "funeral", "battle", "driving car", "city street",
        "restaurant", "gym", "relaxing", "beach", "meditation", "nature sounds"
    ]
}


# ==== 步骤1：裁剪前30秒 ====
def cut_audio(input_path, output_path, duration_sec=30):
    audio = AudioSegment.from_file(input_path)
    first_part = audio[:duration_sec * 1000]
    first_part.export(output_path, format="wav")
    print(f"[+] 音频已裁剪保存为 {output_path}")


# ==== 步骤2：加载模型 ====
def load_yamnet_model():
    print("[*] 正在加载 YAMNet 模型...")
    return hub.load('https://tfhub.dev/google/yamnet/1')


# ==== 步骤3：分类标签 ====
def classify_tag(tag):
    for category, keywords in CATEGORY_MAP.items():
        for keyword in keywords:
            if keyword.lower() in tag.lower():
                return category
    return "Other"


# ==== 步骤4：打标签并分类 ====
def predict_tags(model, wav_file, top_k=10):
    waveform, sr = librosa.load(wav_file, sr=16000)
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)

    scores, embeddings, spectrogram = model(waveform)
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_names = [line.strip() for line in tf.io.gfile.GFile(class_map_path)]

    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    top_indices = np.argsort(mean_scores)[::-1][:top_k]

    results = []
    for i in top_indices:
        tag = class_names[i]
        prob = mean_scores[i]
        category = classify_tag(tag)
        results.append((tag, prob, category))

    print("\n=== 分类标签结果 ===")
    for tag, prob, cat in results:
        print(f"[{cat}] {tag:40s} 概率: {prob:.4f}")

    return results


# ==== 主程序 ====
if __name__ == "__main__":
    # 将MP3转换为WAV格式
    audio = AudioSegment.from_file(input_path)
    wav_path = input_path.replace('.mp3', '.wav')
    audio.export(wav_path, format="wav")
    print(f"[+] 音频已转换为WAV格式: {wav_path}")

    model = load_yamnet_model()
    predict_tags(model, wav_path, top_k=top_k)