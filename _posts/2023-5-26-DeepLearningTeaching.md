---
layout:     post                    # ʹ�õĲ��֣�����Ҫ�ģ�
title:      ���ѧϰ�ڿμ�¼1              # ���� 
subtitle:   2022�˹����ܾ�����������ѧϰ���뽲��
date:       2023-5-26              # ʱ��
author:     zichuana                     # ����
header-img: img/2023-5-26/page.jpg    #��ƪ���±��ⱳ��ͼƬ
catalog: true                       # �Ƿ�鵵
tags:                               #��ǩ
    - �ʼ�
    - flask
---
> ���ڽ��ܴ���ṹ  

��İ�װ����������:)  
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import os
from PIL import Image
```
���Ǹ���ϰ�ߣ������ò����������豸ֻ������CPU  
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```   
�����Ǹ���׼���������������������������б�Ҫ����һ��`transforms`�����������׼������Ҫ��������ݼ������ı任������ֻҪ���С�任��ʵ����`Normalize`��һ����ʹ������Ϊ��������Ҫ�ġ�  

```python
label_train_file = open("./���������/���Ĳ���/train.txt")
labe_test_file = open("./���������/���Ĳ���/test.txt")
pic_path = os.path.join(os.getcwd(), './���������/���Ĳ���/imgdata')
data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize([64, 64])])
```
���ﶼ�ܻ�������һ��һ�ж�txt�����ݿո��֡�  
���ݼ���ǩ�ļ��ṹ��  
![image](/img/2023-5-26/a.png)  
��ͼƬȻ��ʹ�õ���һ���趨�õ�`data_transform`����ͼ��ת����  
���Կ����ҵĴ��������`train_data`��`test_data`����һ�����������Ҫ�õ�����������б�����Ԫ��������ü�ֵ����⣬������ָ����ͼ���`tensor`,��ֵ��ָ���Ǳ�ǩ��`tensor`�������Ƕ��ǩ��mask_len��¼��ǩ����  
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
***������������10 100 50***
```
��`train_data`��`test_data`���뵽DataLoader����׼�������ὫͼƬ�ֳ�`train_data`/`batch_size`�飬ÿ��`batch_size`��ͼƬ���������ȡ��������ڴ�����Դ档  
```python
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=2, shuffle=True, drop_last=True)
```
�����������ѧϰ����ṹ����ʵ����Ҹо�һ��Ҳ���̫���˱��������ֻ����ϣ�����������㶼������ˣ�~~����������~~������������Բ��ᣬ��׼����������ʱ��д�ã�����ֽ�����Ͼ��У��һ��ں������ԭ��  
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
�Ͻ�һ���������һ��NET����net��  
����device������һ�죬�����ѧϰ������device��ע��õĻ������ܻ������������һ����CPUһ����GPU�����⣬��Ȼ���������ǲ�����ֵ�:)  
```python
net = NET(mask_len)
net.to(device)
```
�������ֱ����Ż�������ʧ�����������кܶ��ֵ���ѧԭ�����ܣ�����һ�ѹ�ʽ����x�õ�y�ĺ��壬���ں�������÷���  
```python
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.L1Loss()
```
��������ɹ̶������ʽ�����Ǽ���Ԥ��ֵ����ȷֵ֮��Ĳ�ֵ���������õ���`cosine_similarity`�����������ƶ�������׼ȷ�ʣ�������Ҫ����ͨ������Ҫ�������ί��һ����:)�������ǵ���ǩ����࣬`torch.eq`�Ƚϳ��á�epoch�ǵ���������loss���Ż����������𵽴ٽ����ã������������ǲ��ϸ�����һ��epoch�����ѧϰ�������������Ĵ���Ҫ���������Ľ���**����Ӧ����ѵ���׶ε�ÿһ��`epoch`��ʱ�򶼴���**��  
loss���ǿ��Դ������̶ȣ�������ѧϰ�����������Ĵ���̶Ⱦͻ�Խ��ԽС��ֱ��ƽ�������ʱ���Ϊģ�ʹﵽ������ѵ��������һ�����ƽ��ʱ����׼ȷ�ʣ����Դ������ģ�͵�׼ȷ�ʡ�  
ͨ���ửͼ��˵���������������̳��ţ���ʾÿһ�ε�loss��acc����Ȼ�ñ�������Ҫ������������뻹���㳣�棨��Ա����Ҳ�д����Ҫ��������Ķ��������ⲻ��Ҫ���鷳������Ϊ���ǻ���ѵ��ʱÿһ��epoch����acc��loss��ͨ�����ﻹ���漰����֤��������ȥ��ı���ֻ����ѵ�����Ͳ��Լ��������һ���ϸ���ܡ�  
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
�������������о�������ˣ�����û����ȫ������������ˣ��Ͳ�������ͨ�����ˣ�����22����ոսӴ�����ʱ����ͨ�ĵ�һ���������˽����ô�࣬��Ȼ�����������������븴�Ӻܶ�:)  
�������Ȥ�������濴�ɣ�    
�������׶�ѧϰ����������ʦ����Ƶ[https://space.bilibili.com/1567748478?spm_id_from=333.337.0.0](https://space.bilibili.com/1567748478?spm_id_from=333.337.0.0)  
�ٿ�����֮��Ľ��ܺ���ѧϰ��ѧԭ��Ļ�**ǿ���Ƽ�����**  