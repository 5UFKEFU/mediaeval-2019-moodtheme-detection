import os
# 设置PyTorch内存分配器配置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

import swanlab
from sklearn.metrics import precision_recall_curve
import numpy as np
import argparse
from sklearn.model_selection import KFold
import pandas as pd
import json

from dataloader import get_audio_loader
from solver import Solver

# Set paths
PATH = 'data'
DATA_PATH = f'{PATH}/audio/'
TAG_MAP_PATH = f'{PATH}/tag_map.json'
TRAIN_PATHS = [
    f'{PATH}/moodtheme-train.tsv',
    f'{PATH}/genre-train.tsv',
    f'{PATH}/instrument-train.tsv'
]
VAL_PATHS = [
    f'{PATH}/moodtheme-valid.tsv',
    f'{PATH}/genre-valid.tsv',
    f'{PATH}/instrument-valid.tsv'
]
TEST_PATHS = [
    f'{PATH}/moodtheme-test.tsv',
    f'{PATH}/genre-test.tsv',
    f'{PATH}/instrument-test.tsv'
]

CONFIG = {
        'log_dir': './output',
        'batch_size': 16,
        'num_workers': 4,
        'pin_memory': True,
        'prefetch_factor': 2,
        'persistent_workers': True,
        'n_splits': 5  # 交叉验证折数
    }

def get_labels_to_idx(tag_map_path):
    """从all_tags.txt加载标签映射"""
    with open('data/tags/all_tags.txt', 'r', encoding='utf-8') as f:
        tag_list = [line.strip() for line in f.readlines()]
    
    # 创建标签到索引的映射
    labels_to_idx = {tag: idx for idx, tag in enumerate(tag_list)}
    
    return labels_to_idx, tag_list

def cross_validate():
    """执行交叉验证"""
    # 读取所有训练数据
    train_dfs = [pd.read_csv(path, sep='\t') for path in TRAIN_PATHS]
    kf = KFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=42)
    
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dfs[0])):  # 使用第一个数据集的索引
        print(f"\nStarting fold {fold + 1}/{CONFIG['n_splits']}")
        
        # 创建当前fold的训练和验证数据
        fold_train_paths = []
        fold_val_paths = []
        for df in train_dfs:
            fold_train = df.iloc[train_idx]
            fold_val = df.iloc[val_idx]
            
            # 保存临时数据文件
            train_path = f"{CONFIG['log_dir']}/fold_{fold}_train_{os.path.basename(df.columns[0])}.tsv"
            val_path = f"{CONFIG['log_dir']}/fold_{fold}_val_{os.path.basename(df.columns[0])}.tsv"
            fold_train.to_csv(train_path, sep='\t', index=False)
            fold_val.to_csv(val_path, sep='\t', index=False)
            
            fold_train_paths.append(train_path)
            fold_val_paths.append(val_path)
        
        # 初始化SwanLab实验
        swanlab.init(
            experiment_name=f"multi-tag-detection-fold-{fold}",
            description=f"Cross-validation fold {fold + 1}",
            config=CONFIG
        )
        
        # 训练和验证
        labels_to_idx, tag_list = get_labels_to_idx(TAG_MAP_PATH)
        train_loader1 = get_audio_loader(DATA_PATH, fold_train_paths, 
                                       labels_to_idx, batch_size=CONFIG['batch_size'])
        train_loader2 = get_audio_loader(DATA_PATH, fold_train_paths, 
                                       labels_to_idx, batch_size=CONFIG['batch_size'])
        val_loader = get_audio_loader(DATA_PATH, fold_val_paths, 
                                    labels_to_idx, batch_size=CONFIG['batch_size'], shuffle=False, drop_last=False)
        
        solver = Solver(train_loader1, train_loader2, val_loader, tag_list, CONFIG)
        fold_roc_auc, fold_pr_auc = solver.train()
        
        fold_results.append({
            'fold': fold + 1,
            'roc_auc': fold_roc_auc,
            'pr_auc': fold_pr_auc
        })
        
        # 清理临时文件
        for train_path in fold_train_paths:
            os.remove(train_path)
        for val_path in fold_val_paths:
            os.remove(val_path)
    
    # 计算平均结果
    avg_roc_auc = np.mean([r['roc_auc'] for r in fold_results])
    avg_pr_auc = np.mean([r['pr_auc'] for r in fold_results])
    
    print("\nCross-validation Results:")
    print(f"Average ROC AUC: {avg_roc_auc:.4f}")
    print(f"Average PR AUC: {avg_pr_auc:.4f}")
    
    # 记录到SwanLab
    swanlab.log({
        "cv/avg_roc_auc": avg_roc_auc,
        "cv/avg_pr_auc": avg_pr_auc
    })
    
    return fold_results

def train():
    # 初始化SwanLab实验
    swanlab.init(
        experiment_name="multi-tag-detection",
        description="Training multi-tag detection model (mood, genre, instrument)",
        config=CONFIG
    )
    
    config = CONFIG
    labels_to_idx, tag_list = get_labels_to_idx(TAG_MAP_PATH)    

    train_loader1 = get_audio_loader(DATA_PATH, TRAIN_PATHS, labels_to_idx, batch_size=config['batch_size'])
    train_loader2 = get_audio_loader(DATA_PATH, TRAIN_PATHS, labels_to_idx, batch_size=config['batch_size'])
    val_loader = get_audio_loader(DATA_PATH, VAL_PATHS, labels_to_idx, batch_size=config['batch_size'], shuffle=False, drop_last=False)
    solver = Solver(train_loader1, train_loader2, val_loader, tag_list, config)
    solver.train()

def predict():
    config = CONFIG
    labels_to_idx, tag_list = get_labels_to_idx(TAG_MAP_PATH)

    test_loader = get_audio_loader(DATA_PATH, TEST_PATHS, labels_to_idx, batch_size=config['batch_size'], shuffle=False, drop_last=False)

    solver = Solver(test_loader,None, None, tag_list, config)
    predictions = solver.test()

    # 分别保存预测结果的不同部分
    np.save(f"{CONFIG['log_dir']}/predictions_array.npy", predictions[0])  # 预测数组
    np.save(f"{CONFIG['log_dir']}/ground_truth.npy", predictions[1])      # 真实标签
    np.save(f"{CONFIG['log_dir']}/roc_auc.npy", predictions[2])          # ROC AUC
    np.save(f"{CONFIG['log_dir']}/pr_auc.npy", predictions[3])           # PR AUC
    
    print(f"Predictions saved successfully in {CONFIG['log_dir']}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'predict', 'cross_validate'], 
                       help='Mode to run: train, predict, or cross_validate')
    args = parser.parse_args()
    
    if args.mode == 'train':
        #Train the data
        train()
        #Predict and create submissions
        predict()
    elif args.mode == 'cross_validate':
        # Run cross-validation
        cross_validate()
    else:
        #Only predict
        predict()