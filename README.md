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

image.png



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
