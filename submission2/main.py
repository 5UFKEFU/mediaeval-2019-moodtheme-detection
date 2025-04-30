import os
# 设置PyTorch内存分配器配置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

from sklearn.metrics import precision_recall_curve
import numpy as np

from dataloader import get_audio_loader
from solver import Solver

# Set paths
PATH = '../data'
DATA_PATH = f'{PATH}/mediaeval-2019-jamendo/'
LABELS_TXT = f'{PATH}/moodtheme_split.txt'
TRAIN_PATH = f'{PATH}/autotagging_moodtheme-train.tsv'
VAL_PATH = f'{PATH}/autotagging_moodtheme-valid.tsv'
TEST_PATH = f'{PATH}/autotagging_moodtheme-test.tsv'

CONFIG = {
        'log_dir': './output',
        'batch_size': 16,
        'num_workers': 4,
        'pin_memory': True,
        'prefetch_factor': 2,
        'persistent_workers': True
    }

def get_labels_to_idx(labels_txt):
    labels_to_idx = {}
    tag_list = []
    with open(labels_txt) as f:
        lines = f.readlines()

    for i,l in enumerate(lines):
        tag_list.append(l.strip())
        labels_to_idx[l.strip()] = i

    return labels_to_idx, tag_list

def train():
    config = CONFIG
    labels_to_idx, tag_list = get_labels_to_idx(LABELS_TXT)    

    train_loader1 = get_audio_loader(DATA_PATH, TRAIN_PATH, labels_to_idx, batch_size=config['batch_size'])
    train_loader2 = get_audio_loader(DATA_PATH, TRAIN_PATH, labels_to_idx, batch_size=config['batch_size'])
    val_loader = get_audio_loader(DATA_PATH, VAL_PATH, labels_to_idx, batch_size=config['batch_size'], shuffle=False, drop_last=False)
    solver = Solver(train_loader1,train_loader2, val_loader, tag_list, config)
    solver.train()

def predict():
    config = CONFIG
    labels_to_idx, tag_list = get_labels_to_idx(LABELS_TXT)

    test_loader = get_audio_loader(DATA_PATH, TEST_PATH, labels_to_idx, batch_size=config['batch_size'], shuffle=False, drop_last=False)

    solver = Solver(test_loader,None, None, tag_list, config)
    predictions = solver.test()

    # 分别保存预测结果的不同部分
    np.save(f"{CONFIG['log_dir']}/predictions_array.npy", predictions[0])  # 预测数组
    np.save(f"{CONFIG['log_dir']}/ground_truth.npy", predictions[1])      # 真实标签
    np.save(f"{CONFIG['log_dir']}/roc_auc.npy", predictions[2])          # ROC AUC
    np.save(f"{CONFIG['log_dir']}/pr_auc.npy", predictions[3])           # PR AUC
    
    print(f"Predictions saved successfully in {CONFIG['log_dir']}")


if __name__=="__main__":

    #Train the data
    train()

    #Predict and create submissions
    predict()