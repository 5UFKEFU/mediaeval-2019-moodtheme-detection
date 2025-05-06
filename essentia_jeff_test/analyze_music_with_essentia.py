import os
import sys
import re
import numpy as np

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
essentia._setDebugLevel(0,0)  # 完全静音，依靠 WarnOnce 打印首条错误即可
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

# 定义模型文件路径和对应的输出节点
model_configs = {
    'mood_happy': {
        'file': 'essentia_models/mood_happy-musicnn-msd-2.pb',
        'output': 'model/Sigmoid'
    },
    'mood_sad': {
        'file': 'essentia_models/mood_sad-musicnn-msd-2.pb',
        'output': 'model/Sigmoid'
    },
    'mood_relaxed': {
        'file': 'essentia_models/mood_relaxed-musicnn-msd-2.pb',
        'output': 'model/Sigmoid'
    },
    'mood_acoustic': {
        'file': 'essentia_models/mood_acoustic-musicnn-msd-2.pb',
        'output': 'model/Sigmoid'
    },
    'mood_aggressive': {
        'file': 'essentia_models/mood_aggressive-musicnn-msd-2.pb',
        'output': 'model/Sigmoid'
    },
    'mood_electronic': {
        'file': 'essentia_models/mood_electronic-musicnn-msd-2.pb',
        'output': 'model/Sigmoid'
    },
    'danceability': {
        'file': 'essentia_models/danceability-musicnn-msd-2.pb',
        'output': 'model/Sigmoid'
    },
    'gender': {
        'file': 'essentia_models/gender-musicnn-msd-2.pb',
        'output': 'model/Sigmoid'
    },
    'mood_party': {
        'file': 'essentia_models/mood_party-musicnn-msd-2.pb',
        'output': 'model/Sigmoid'
    },
    'moods_mirex': {
        'file': 'essentia_models/moods_mirex-musicnn-msd-1.pb',
        'output': 'model/Softmax'
    },
    'genre_tzanetakis': {
        'file': 'essentia_models/genre_tzanetakis-musicnn-msd-2.pb',
        'output': 'model/Sigmoid'
    },
    'tonal_atonal': {
        'file': 'essentia_models/tonal_atonal-musicnn-msd-2.pb',
        'output': 'model/Sigmoid'
    },
    'voice_instrumental': {
        'file': 'essentia_models/voice_instrumental-musicnn-msd-2.pb',
        'output': 'model/Sigmoid'
    }
}

def analyze_file(filepath):
    try:
        print(f"正在分析文件: {filepath}")
        # 加载音频文件，重采样到16kHz
        audio = MonoLoader(filename=filepath, sampleRate=16000, resampleQuality=4)()
        print(f"音频加载成功，长度: {len(audio)}")
        
        # 初始化模型
        models = {}
        for tag, config in model_configs.items():
            models[tag] = TensorflowPredictMusiCNN(
                graphFilename=config['file'],
                output=config['output']
            )
        
        result = {}
        for tag, model in models.items():
            print(f"处理标签: {tag}")
            prediction = model(audio)
            
            # 如果预测结果是矩阵，取时间维度的平均值
            if isinstance(prediction, np.ndarray):
                if prediction.ndim > 1:
                    prediction = np.mean(prediction, axis=0)
                if prediction.size > 1:
                    prediction = prediction[0]
            elif isinstance(prediction, list):
                prediction = np.mean(prediction)
            
            # 确保结果在0-1之间
            if tag == 'moods_mirex':
                # Softmax输出，取最大值
                result[tag] = float(np.max(prediction))
            else:
                # Sigmoid输出，直接使用
                result[tag] = float(prediction)
            
            print(f"{tag}: {result[tag]:.3f}")
            
        return result
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return {}

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