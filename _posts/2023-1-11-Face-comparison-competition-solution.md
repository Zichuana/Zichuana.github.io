---
layout:     post                    # 使用的布局（不需要改）
title:      计挑赛人脸比对竞赛解决方法               # 标题 
subtitle:   第四届计算机能力挑战赛大数据与人工智能赛道
date:       2023-1-11              # 时间
author:     zichuana                     # 作者
header-img: img/2023-1-11/page.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 深度学习
---
> 源码上传至github[https://github.com/Zichuana/Face-comparison-competition-solution](https://github.com/Zichuana/Face-comparison-competition-solution)
> 该解决方案获得华东赛区一等奖，国赛二等奖

## 1.整个赛题的解决思路以及用到的数据处理方法
### 解决思路与引用介绍
*解决第四届计算机能力挑战赛大数据与人工智能赛题*  
搭建VGG孪生网络训练所给数据集得到模型`train_net.pth`，并使用Facenet-PyTorch框架训练适应于赛题的模型`facenet_mobilenet_test.pth`，预测时按一定比例就行模型融合，分别所使用到的预训练模型`VGG Face`与`facenet_mobilenet.pth`，模型选取并参考代码于[https://www.kaggle.com/code/anki08/modified-siamese-network-pytorch](https://www.kaggle.com/code/anki08/modified-siamese-network-pytorch)和[https://github.com/bubbliiiing/facenet-pytorch](https://github.com/bubbliiiing/facenet-pytorch)。  
其中在`init_data/train`下`data`存放竞赛提供的数据集，`dataset`下存放扩增的数据集，扩增的数据集节选自`CASIA-WebFace`，由[https://github.com/bubbliiiing/facenet-pytorch](https://github.com/bubbliiiing/facenet-pytorch)并对其就行一定的处理，以适应竞赛。  
扩增的数据集百度网盘下载链接:[https://pan.baidu.com/s/1qMxFR8H_ih0xmY-rKgRejw](https://pan.baidu.com/s/1qMxFR8H_ih0xmY-rKgRejw)  
提取码: bcrq。  
竞赛数据集自留了一份上传在google硬盘[https://drive.google.com/drive/folders/1WpK2yDjHdPS-MzC6sA3HBCLTZqgMlymG?usp=share_link](https://drive.google.com/drive/folders/1WpK2yDjHdPS-MzC6sA3HBCLTZqgMlymG?usp=share_link)
### 数据处理方法
- 对`data`下数据
先对图像去除黑色边框，对`a`类图像直接就行高斯滤波处理噪声，对`b`类图像先添加椒盐噪声再就行高斯滤波处理。分别对`a`图`b`图就行左右翻转，组合，将竞赛方提供的数据集扩增4倍。
- 对`dataset`下数据
在每一编号下的图像中选取20%对其增加椒盐噪声并就行高斯滤波处理，重新按原始名称保存（处理后仍在该文件夹下）。
## 2.使用的算法模型简要介绍
- `train_net.pth` 模型一
直接获取每个编号的`a`图和`b`图以及标签，对图像就行处理后进入网络进行训练。对结果进行0，1分类，反馈网络，并获得准确率。
网络结构输出如下：
```python
Vgg_face_dag(
  (conv1_1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu1_1): ReLU(inplace=True)
  (conv1_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu1_2): ReLU(inplace=True)
  (pool1): MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
  (conv2_1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu2_1): ReLU(inplace=True)
  (conv2_2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu2_2): ReLU(inplace=True)
  (pool2): MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
  (conv3_1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu3_1): ReLU(inplace=True)
  (conv3_2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu3_2): ReLU(inplace=True)
  (conv3_3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu3_3): ReLU(inplace=True)
  (pool3): MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
  (conv4_1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu4_1): ReLU(inplace=True)
  (conv4_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu4_2): ReLU(inplace=True)
  (conv4_3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu4_3): ReLU(inplace=True)
  (pool4): MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
  (conv5_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu5_1): ReLU(inplace=True)
  (conv5_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu5_2): ReLU(inplace=True)
  (conv5_3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu5_3): ReLU(inplace=True)
  (pool5): MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
  (fc6): Linear(in_features=25088, out_features=4096, bias=True)
  (relu6): ReLU(inplace=True)
  (dropout6): Dropout(p=0.5, inplace=False)
  (fc7): Linear(in_features=4096, out_features=4096, bias=True)
  (relu7): ReLU(inplace=True)
  (dropout7): Dropout(p=0.5, inplace=False)
  (fc8): Linear(in_features=4096, out_features=2622, bias=True)
)
Classification(
  (fc1): Sequential(
    (0): Linear(in_features=2622, out_features=1024, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=1024, out_features=512, bias=True)
    (3): ReLU(inplace=True)
    (4): Linear(in_features=512, out_features=256, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=256, out_features=128, bias=True)
    (7): ReLU(inplace=True)
    (8): Linear(in_features=128, out_features=2, bias=True)
  )
)
```
- `facenet_mobilenet_test.pth` 模型二
分别取两张同一个图片与一张另一个人的图片，分别作为标签同一个人与非同一人，在使用框架的基础上就行一定的预处理进入`mobilenet`就行训练。  
网络结构输出如下：
```python
Facenet(
  (backbone): mobilenet(
    (model): MobileNetV1(
      (stage1): Sequential(
        (0): Sequential(
          (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6()
        )
        (1): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6()
          (3): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6()
        )
        (2): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6()
          (3): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6()
        )
        (3): Sequential(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6()
          (3): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6()
        )
        (4): Sequential(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6()
          (3): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6()
        )
        (5): Sequential(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6()
          (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6()
        )
      )
      (stage2): Sequential(
        (0): Sequential(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6()
          (3): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6()
        )
        (1): Sequential(
          (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6()
          (3): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6()
        )
        (2): Sequential(
          (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6()
          (3): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6()
        )
        (3): Sequential(
          (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6()
          (3): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6()
        )
        (4): Sequential(
          (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6()
          (3): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6()
        )
        (5): Sequential(
          (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6()
          (3): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6()
        )
      )
      (stage3): Sequential(
        (0): Sequential(
          (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=512, bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6()
          (3): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6()
        )
        (1): Sequential(
          (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024, bias=False)
          (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6()
          (3): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6()
        )
      )
    )
  )
  (avg): AdaptiveAvgPool2d(output_size=(1, 1))
  (Dropout): Dropout(p=0.5, inplace=False)
  (Bottleneck): Linear(in_features=1024, out_features=128, bias=False)
  (last_bn): BatchNorm1d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  (classifier): Linear(in_features=128, out_features=10575, bias=True)
)
```
- 对模型一二就行1:99的融合，高于二者未融合的结果，在比赛评判系统测试获得0.89~0.92之间的成绩。（国赛与区域赛相同run，结果不同，具体不记得了…）
## 3.整个模型的实现过程的详细步骤说明
### code文件说明
```html
code
│  dataloader.py
│  model.py
│  predict.py
│  train.py
│  txt_annotation.py
│  utils.py
│  run.py
│
└─model
        facenet_mobilenet_test.pth
        facenet_mobilenet.pth
        train_net.pth
        VGG Face
```
`model.py`为网络搭建文件。  
`train.py`为训练文件。  
`dataloader.py`和`utils.py`内包含部分训练预测共同使用的函数。  
`predict.py`为自测试准确率文件。  
`txt_annotation.py`为对dataset内训练集进行预处理文件。  
`run.py`为提交分数最高的run文件，注意该文件不符合`my_project`文件路径，无法正常运行。  
- 模型介绍
`VGG Face`为模型一的预训练权重  
`facenet_mobilenet.pth`为模型二的预训练权重  
模型放到google硬盘[https://drive.google.com/drive/folders/1Q3vKBLWDWLSz5uhOhchJj03NVpOjAX1g?usp=share_link](https://drive.google.com/drive/folders/1Q3vKBLWDWLSz5uhOhchJj03NVpOjAX1g?usp=share_link)
### 实现步骤
在基础库上`pip install -r requirements.txt`安装一致版本的pytorch，opencv，以及细小工具不含有其它库(如果复现时发现还缺少什么其它的都是可以直接pip安装的，出现缺少的问题原因在环境里还包含其它库，requirements是手动输入的，但大概率与版本无关)。  
将`init_data/train/datasets`下存放扩增数据集，网盘连接如前所示，将内容解压到datasets内，运行`txt_annotation.py`文件获得标签文件`cls_train.txt`在`init_data/train`目录下，这里使用的是绝对路径。运行`train.py`在各版版本匹配的情况下即可实现。
## 4.算法运行（环境）说明
可支持环境CUDA 11.3，python>=3.7。  
自电脑使用环境cpu，python=3.10。  
算力环境CUDA 11.3，python=3.8。