---
layout:     post                    # 使用的布局（不需要改）
title:      安徽省大数据与人工智能竞赛人工智能软件赛-深度学习               # 标题 
subtitle:   网络赛与现场赛第四题深度学习代码 #副标题
date:       2022-11-26              # 时间
author:     zichuana                     # 作者
header-img: img/2022-11-26/page.png    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 碎碎念
    - 深度学习
---
> :expressionless:	

## 网络赛
### 题目
“第四部分”文件夹为本题的数据文件，含有100张交通场景监控图片及其对应的训练和测试标签文件trainlabels.txt和testlabels.txt。标签文件第1列表示图片路径，第2-8列表示其对应的图片中是否存在对应的车型，若存在则用1表示，若不存在则用0表示。请根据要求利用卷积神经网络实现图片的多标签分类功能。
要求与说明：
（1）本体旨在考察参赛者使用深度学习库解决具体计算机视觉问题的完整流程，不以测试集准确率作为评分标准；
（2）输入网络的图片数据尺寸统一缩放到64*64*3；
（3）网络的输出层为全连接神经网络，其中的每一个神经元对应一个物体类别，中间层自由搭建，但要保证维度匹配、代码可运行；
（4）使用Adam优化器；
（5）使用L1损失函数；
（6）代码能正常运行，输出每一次或每一轮迭代的损失值情况；
（7）使用深度学习库PyTorch或Tensorflow均可；
（8）在训练集进行训练，在测试集进行测试；
（9）需将所有代码及成功训练运行截图放入比赛环境中本题对应的提交目录文件夹（user/Q4文件夹）。
### 代码
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.utils.data.dataloader as Data
import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
label_train_file = open("./数据与代码/第四部分/trainlabels.txt")
label_test_file = open("./数据与代码/第四部分/testlabels.txt")
pic_path = os.path.join(os.getcwd(), './数据与代码/第四部分/')

data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize([64, 64]),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

train_data = []
train_data_len = 0
for i, v in enumerate(label_train_file):
    mid = v.strip().split(' ')
    img_file = mid[0]
    label = mid[1:len(mid)]
    label = [int(i) for i in label]
    # print(label)
    img = Image.open(os.path.join(pic_path, img_file))
    img = data_transform(img)
    label = torch.tensor(label)
    # print(img)
    train_data.append(tuple([img, label]))
    train_data_len += 1
# print(train_data)

test_data = []
test_data_len = 0
for i, v in enumerate(label_test_file):
    mid = v.strip().split(' ')
    img_file = mid[0]
    label = mid[1:len(mid)]
    label = [int(i) for i in label]
    img = Image.open(os.path.join(pic_path, img_file))
    img = data_transform(img)
    label = torch.tensor(label)
    test_data.append(tuple([img, label]))
    test_data_len += 1
    mask_len = len(v.strip().split(' ')[1:len(v.strip().split(' '))])
# print(len)
train_loader = Data.DataLoader(dataset=train_data, batch_size=16, shuffle=True, drop_last=True)
test_data = Data.DataLoader(dataset=test_data, batch_size=2, shuffle=True, drop_last=True)  # batch_size???


# print(train_loader)


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
            nn.ReLU()
        )
        self.fc = nn.Linear(24 * 3 * 3, len)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output = self.sigmoid(x)
        return output


net = NET(mask_len)
print(net)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.L1Loss()
for epoch in range(80):
    train_loss = 0.0
    for index, (batch_x, batch_y) in enumerate(train_loader):
        # print(batch_x)
        # print(batch_y)
        # exit()
        out = net(batch_x.to(device))
        # print(batch_y)
        # print(out)
        # loss = loss_func(out, multilabel_generate(batch_y, mask_len))  # / batch_x.shape[0]
        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch:{} Train Loss:{}'.format(epoch + 1, train_loss / train_data_len))
    net.eval()

with torch.no_grad():
    TP = 0  # y_true与y_pred中同时为1的个数
    TN = 0  # y_true中为0但是在y_pred中被识别为1的个数
    FP = 0  # y_true中为1但是在y_pred中被识别为0的个数
    FN = 0  # y_true与y_pred中同时为0的个数
    for index, (batch_x, batch_y) in enumerate(test_data):
        pre = net(batch_x.to(device))
        for x, y in zip(pre, batch_y):
            for i, j in enumerate(x):
                if j.item() > 0.5:
                    j = 1
                else:
                    j = 0
                # print(j, batch_y[0][i])
                if y[i].item() == 1 and j == 1:
                    TP += 1
                if y[i].item() == 1 and j == 0:
                    FP += 1
                if y[i].item() == 0 and j == 0:
                    FN += 1
                if y[i].item() == 0 and j == 1:
                    TN += 1
    acc = (TP + FN) / (TP + TN + FN + FP)
    print("acc:", acc)

```
部分运行结果
```python
Epoch:68 Train Loss:0.008207786745495266
Epoch:69 Train Loss:0.008494619362884098
Epoch:70 Train Loss:0.007947015100055271
Epoch:71 Train Loss:0.008119823204146491
Epoch:72 Train Loss:0.008102810465627247
Epoch:73 Train Loss:0.008297939101854961
Epoch:74 Train Loss:0.008286461068524254
Epoch:75 Train Loss:0.008195203708277808
Epoch:76 Train Loss:0.008094929820961423
Epoch:77 Train Loss:0.008372044232156541
Epoch:78 Train Loss:0.008071828716331058
Epoch:79 Train Loss:0.00809686283270518
Epoch:80 Train Loss:0.00870207084549798
acc: 0.8285714285714286
```
## 现场赛
### 题目
“第四部分”文件夹为本题的数据文件，含有150张人脸图片及其对应的训练和测试标签文件train.txt和test.txt。标签文件第1列表示图片文件名，第2-11列表示其对应的人脸图片的五个关键点的坐标（x1,y1,x2,y2,…x5,y5）。请根据要求利用卷积神经网络实现人脸图片的关键点定位功能。
要求与说明：
（1）本体旨在考察参赛者使用深度学习库解决具体计算机视觉问题的完整流程，不以测试集准确率作为评分标准；
（2）输入网络的图片数据尺寸统一缩放到`64*64*3`；
（3）网络的输出层为全连接神经网络，共 10 个神经元，分别对应五个关键点的坐标，中间层自由搭建，但要保证维度匹配、代码可运行；
（4）使用 Adam 优化器；
（5）使用 L1 损失函数；
（6）代码能正常运行，输出每一次或每一轮迭代的损失值情况；
（7）使用深度学习库 PyTorch 或 Tensorflow 均可；
（8）在训练集进行训练，在测试集进行测试；
（9）需将所有代码及成功训练运行截图放入比赛环境中本题对应的提交目录文件夹
（user/Q4 文件夹）。
### 代码
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
# import torch.utils.data as Data
import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import numpy as np
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
label_train_file = open("./数据与代码/第四部分/train.txt")
labe_test_file = open("./数据与代码/第四部分/test.txt")
pic_path = os.path.join(os.getcwd(), './数据与代码/第四部分/imgdata')
data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize([64, 64])])
train_data = []
train_data_len = 0
for i, v in enumerate(label_train_file):
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
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=2, shuffle=True, drop_last=True)


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


net = NET(mask_len)
print(net)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.L1Loss()
for epoch in range(100):
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
部分运行结果
```python
Epoch:87 Train Loss:0.1991818380355835
Epoch:88 Train Loss:0.16999389171600343
Epoch:89 Train Loss:0.18029738903045656
Epoch:90 Train Loss:0.19433170080184936
Epoch:91 Train Loss:0.1881941056251526
Epoch:92 Train Loss:0.1744999933242798
Epoch:93 Train Loss:0.15164695739746092
Epoch:94 Train Loss:0.15582487106323242
Epoch:95 Train Loss:0.1559641432762146
Epoch:96 Train Loss:0.13775283098220825
Epoch:97 Train Loss:0.1374074649810791
Epoch:98 Train Loss:0.13810728073120118
Epoch:99 Train Loss:0.13806066751480103
Epoch:100 Train Loss:0.13971636056900025
acc 0.9995065593719482
```
网络赛和现场赛网络结构基本相同，但现场赛不需要使用逻辑回归Sigmoid函数进行结果转换。此外现场赛最后准确率计算方法，所学里一时没有想到用什么，就使用的是余弦相似度，可以扯出道理，但或多或少带点毛病在……  
竞赛体验一般，我负责写深度学习和opencv三道题，三个人线上光使用U盘互传文件归纳答题卡就花了四十分钟（U盘有点bug，时不时文件丢失），以及队友拉分最大的一题代码没有跑出来:sob:，很失望也不敢直说（啊不对，卷王表示非常失望！），毕竟谁都有失误，但是果然还是只有我和cc最靠谱嘤嘤嘤……  
关于队友太信任也不行，不信任又把自己累成修:dog:，嘤嘤嘤嘤嘤嘤……  
遇到这个情况我也只能嘤嘤嘤嘤嘤嘤:cry: