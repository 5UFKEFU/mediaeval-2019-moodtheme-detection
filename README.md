增加了一些参数：
继续训练
python main.py --mode train --resume output/model_epoch_60.pth

检验模式
python main.py --mode predict

正常训练
python main.py --mode train

小批量训练
python main.py --mode train --test
训练集：只使用前100条数据
验证集：只使用前20条数据
测试集：只使用前20条数据

测试验证准确率
使用python.exe .\validate_model.py --label_type instrument
得到以下的类似的结果：
总体验证结果:
平均精确率: 0.25 | 平均召回率: 0.24 | 平均F1分数: 0.24
处理曲目数量: 100

结果显示训练效果很一般，猜中率很有限，训练了几天了，浪费了很多电，实际上效果很差，可能我也理解了这个活动官方举办了3年了，看后面实在没有什么进展，就放弃了。
音乐看来太主观了，不适合机器学习吧。



以下是原项目中的文件说明
================================================================================

# Solution to the "MediaEval - The 2019 Emotion and Themes in Music using Jamendo" task

This repository contains our solution to the 2019 Emotion and Themes in Music using Jamendo task, part of [MediaEval 2019](http://www.multimediaeval.org/mediaeval2019/).

## Introduction

For details about the task, please follow the links:
- http://www.multimediaeval.org/mediaeval2019/music/
- https://multimediaeval.github.io/2019-Emotion-and-Theme-Recognition-in-Music-Task/

The solution is described in the [report](MediaEval_19_paper_35.pdf).

## Submissions

Please check the submission folder for submissions. There are two submissions made:

- submission1 - MobilenetV2 + Data Augmentation
- submission2 - MobilenetV2 + Data Augmentation + Self Attention

## Results

Please see https://multimediaeval.github.io/2019-Emotion-and-Theme-Recognition-in-Music-Task/results.

## Citing and license

Authors:

- [Manoj Sukhavasi](https://github.com/manojsukhavasi)
- [Sainath Adapa](https://github.com/sainathadapa)

This work is licensed under the [MIT License](LICENSE).
