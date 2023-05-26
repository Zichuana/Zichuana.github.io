---
layout:     post                    # 使用的布局（不需要改）
title:      深度学习授课记录1              # 标题 
subtitle:   2022人工智能竞赛软件组深度学习代码讲解
date:       2023-5-26              # 时间
author:     zichuana                     # 作者
header-img: img/2023-5-26/page.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 笔记
    - flask
---
> 本节介绍代码结构  

库的安装跳过不介绍:)  
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import os
from PIL import Image
```
这是个好习惯，比赛用不到，比赛设备只给得起CPU  
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```   
这里是各种准备工作，创建对象声明变量，有必要介绍一下`transforms`你可以在这里准备好你要对你的数据集所做的变换，比赛只要求大小变换，实际上`Normalize`归一化的使用我认为才是最重要的。  

```python
label_train_file = open("./数据与代码/第四部分/train.txt")
labe_test_file = open("./数据与代码/第四部分/test.txt")
pic_path = os.path.join(os.getcwd(), './数据与代码/第四部分/imgdata')
data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize([64, 64])])
```
这里都很基础啊，一行一行读txt，依据空格拆分。  
数据集标签文件结构：  
![image](/img/2023-5-26/a.png)  
读图片然后使用到上一步设定好的`data_transform`就行图像转换。  
可以看到我的代码里面的`train_data`和`test_data`，这一步骤里面的需要得到东西，这个列表里面元素你可以用键值对理解，“键”指的是图像的`tensor`,“值”指的是标签的`tensor`，这里是多标签，mask_len记录标签数。  
```python
train_data = []
train_data_len = 0
for i, v in enumerate(label_traian_file):
    mid = v.strip().split()
    img_file = mid[0]
    labels = mid[1:len(mid)]
    label = [int(i) for i in labels]
    img = Image.open(os.path.join(pic_path, img_file))
    img = data_transform(img)
    label = torch.tensor(label)
    train_data.append(tuple([img, label]))
    train_data_len += 1
test_data = []
test_data_len = 0
for i, v in enumerate(labe_test_file):
    mid = v.strip().split()
    img_file = mid[0]
    labels = mid[1:len(mid)]
    label = [int(i) for i in labels]
    img = Image.open(os.path.join(pic_path, img_file))
    img = data_transform(img)
    label = torch.tensor(label)
    test_data.append(tuple([img, label]))
    # print(test_data[0])
    test_data_len += 1
    mask_len = len(v.strip().split()[1:len(v.strip().split())])
print(mask_len, train_data_len, test_data_len)
***这里输出结果是10 100 50***
```
将`train_data`和`test_data`输入到DataLoader中做准备，它会将图片分成`train_data`/`batch_size`块，每块`batch_size`张图片，这里多少取决于你的内存或者显存。  
```python
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=2, shuffle=True, drop_last=True)
```
接下来是深度学习网络结构（其实这个我感觉一点也不深，太深了比赛数据又会过拟合，就我这个几层都过拟合了，~~比赛很正规~~），这里你可以不会，在准备网络赛的时候写好，带上纸质资料就行，我会在后面介绍原理。  
```python
class NET(nn.Module):
    def __init__(self, len):
        super(NET, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        self.fc = nn.Linear(24 * 3 * 3, len)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        # output = self.sigmoid(x)
        return output
```
严谨一点这叫声明一个NET对象net。  
看到device我想提一嘴，在你的学习过程中device不注意好的话，可能会出现两个张量一个在CPU一个在GPU的问题，当然比赛电脑是不会出现的:)  
```python
net = NET(mask_len)
net.to(device)
```
这两个分别是优化器和损失函数，方法有很多种但数学原理不介绍，就是一堆公式输入x得到y的含义，会在后面介绍用法。  
```python
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.L1Loss()
```
你可以理解成固定代码格式，但是计算预测值和正确值之间的差值，这里我用的是`cosine_similarity`计算余弦相似度来代表准确率，根据需要灵活变通，不需要多合理评委不一定懂:)，假设是单标签多分类，`torch.eq`比较常用。epoch是迭代次数，loss和优化器在这里起到促进作用，可以理解成他们不断告诉上一次epoch，这次学习出现这样那样的错误，要这样那样改进！**他们应该在训练阶段的每一次`epoch`的时候都存在**。  
loss就是可以代表错误程度，假设它学习能力不错，它的错误程度就会越来越小，直到平缓，这个时候称为模型达到收敛，训练结束，一般这段平缓时区的准确率，可以代表这个模型的准确率。  
通常会画图（说不定还会出个解读教程嗯）表示每一次的loss和acc，当然该比赛不需要，下面这个代码还不算常规（针对比赛我不写不需要我们输出的东西，避免不必要的麻烦），因为我们会在训练时每一次epoch计算acc和loss，通常这里还会涉及到验证集，但是去年的比赛只存在训练集和测试集，后面我会详细介绍。  
```python
for epoch in range(10):
    train_loss = 0.0
    for index, (batch_x, batch_y) in enumerate(train_loader):
        out = net(batch_x.to(device))
        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch:{} Train Loss:{}'.format(epoch + 1, train_loss / train_data_len))

net.eval()
with torch.no_grad():
    acc = 0.0
    for index, (batch_x, batch_y) in enumerate(test_loader):
        pre = net(batch_x.to(device))
        similar = torch.cosine_similarity(batch_y.to(device), pre)
        acc += similar.sum().item()
    print("acc", acc / test_data_len)
```
看到这里如果你感觉你好像懂了，但是没有完全懂，这个好像懂了，就差不多可以跑通代码了，我在22年初刚刚接触它的时候，跑通的第一个代码差不多了解的这么多，当然比这个比赛的这个代码复杂很多:)  
如果感兴趣就往后面看吧！    
我修炼阶段学习的是李沐老师的视频[https://space.bilibili.com/1567748478?spm_id_from=333.337.0.0](https://space.bilibili.com/1567748478?spm_id_from=333.337.0.0)  
再看完我之后的介绍后还想学习数学原理的话**强烈推荐！！**  