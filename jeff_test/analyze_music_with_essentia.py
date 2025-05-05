import os
import sys
import re

# --- Show at most one "No network created ..." warning ---
_warn_re = re.compile(r"No network created, or last created network has been deleted")

class WarnOnce:
    """Allow only the first matching warning line to pass through stderr."""
    def __init__(self, original):
        self.original = original
        self.seen = False
    def write(self, msg):
        if _warn_re.search(msg):
            if not self.seen:
                self.seen = True
                self.original.write(msg)
        else:
            self.original.write(msg)
    def flush(self):
        self.original.flush()

sys.stderr = WarnOnce(sys.stderr)
# ----------------------------------------------------------

from essentia.standard import MonoLoader, TensorflowPredictMusiCNN

# Silence Essentia INFO and WARNING messages
import essentia
# 0 = silent, 1 = error, 2 = warning, 3 = info
essentia._setDebugLevel(0)  # 完全静音，依靠 WarnOnce 打印首条错误即可
# 设置你要分析的目录
MUSIC_DIR = './test_music'
# OUTPUT_CSV = 'analysis_results.csv'

# 支持的音频格式
SUPPORTED_EXTENSIONS = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']

# 你要提取的标签
TAGS = [
    # Original tags
    'mood_happy', 'mood_sad', 'mood_relaxed',
    'mood_acoustic', 'mood_aggressive',
    'mood_electronic', 'danceability', 'gender',
    # Additional models
    'mood_party', 'moods_mirex',
    'genre_dortmund', 'genre_electronic',
    'genre_rosamerica', 'genre_tzanetakis',
    'tonal_atonal', 'voice_instrumental',
    'urbansound8k', 'fs_loop_ds'
]

# 标签与模型文件的映射（统一命名规则）
model_files = {
    tag: f"essentia_models/{tag}-musicnn-msd-2.pb" for tag in TAGS
}

# 加载模型一次（共享）
# reloadGraph=False  ->  仅在算法实例化时构建网络；后续调用复用同一网络
models = {
    tag: TensorflowPredictMusiCNN(
        graphFilename=model_files[tag],
        output='model/Sigmoid'
    ) for tag in TAGS
}

def analyze_file(filepath):
    try:
        audio = MonoLoader(filename=filepath)()
        result = {}
        for tag, model in models.items():
            prediction = model(audio)  # 可能返回 (frames, tags) 或 (tags,)
            if prediction.ndim == 2:
                prediction = prediction.mean(axis=0)  # 每个标签取时间平均
            prediction = prediction.flatten()

            if prediction.size == 1:
                result[tag] = round(float(prediction[0]), 3)
            elif prediction.size == 2:
                result[tag] = round(float(prediction[1]), 3)
            else:
                result[tag] = round(float(prediction.mean()), 3)
        return result
    except Exception as e:
        print(f"[ERROR] {filepath} - {e}")
        return None

def main():
    for root, _, files in os.walk(MUSIC_DIR):
        for file in files:
            if file.startswith('._'):
                continue  # 跳过 macOS 资源派生文件
            if any(file.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                path = os.path.join(root, file)
                analysis = analyze_file(path)
                summary = ", ".join(f"{k}:{v:.3f}" for k, v in analysis.items())
                print(f"{file} → {summary}")

if __name__ == '__main__':
    main()