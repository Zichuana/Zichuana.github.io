---
layout:     post                    # 使用的布局（不需要改）
title:      深度学习授课记录3              # 标题 
subtitle:   2022人工智能竞赛软件组深度学习训练过程讲解
date:       2023-5-26              # 时间
author:     zichuana                     # 作者
header-img: img/2023-5-26/page.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 深度学习
---
> 本节介绍深度学习网络模型的训练过程  

22年人工智能赛题只要求输出loss，所以没有输出训练时每次epoch准确率，这个还是要有吧，假设训练集和验证集的acc输出的话，是可以很好地判断模型是否拟合，比赛并不要求这一点。  
但是这个我觉得还是很有必要提醒一下，避免因为这个比赛导致了一些误区。  
通常训练一个模型是需要划分成，训练集，验证集和测试集的。在测试集和验证集上训练，在测试集上测试，两者都可以作为评价模型的标准。  
关于这一点的介绍，我决定写注释来介绍代码，原理感兴趣就听李沐老师吧！  
```python
train_steps = len(train_loader)  # 获取每一次epoch内传入图像的泼数，一泼16张
net = NET(mask_len)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # 创建一个优化器，在这里选用你模型训练阶段所使用的优化器
loss_func = torch.nn.L1Loss()  # 创建一个损失函数
for epoch in range(10):   # 这里epoch 10次意思一下
    train_loss = 0.0  # 指的是每一次epoch的loss和acc 不要和下面index混淆了
    train_acc = 0.0
    for index, (batch_x, batch_y) in enumerate(train_loader):  # 将打包在train_loader的图像送进去
        out = net(batch_x.to(device))  # 可以理解成网络学习的结果
        loss = loss_func(out, batch_y)  # 使用损失函数计算这一次index的损失值
        optimizer.zero_grad()  # 优化器清零
        loss.backward()  # 损失结果反向传播
        optimizer.step()  # 优化器使用
        train_loss += loss.item()  # 加到train_loss里面，再用train_loss除以传入图像的泼数就是这一次epoch的loss平均值了，用这个值
        similar = torch.cosine_similarity(out, batch_y.to(device))  # 计算相似度
        train_acc += similar.sum().item()  # 将相似度整合到总准确率中，这里除与总的图像张数，得到每张图像的平均准确率
    print('Epoch:{} Train Loss:{} acc:{}'.format(epoch + 1, train_loss / train_steps, train_acc / train_data_len))

    net.eval()  # 不在反向传播了，这里是开始测试或者验证的标志
    val_acc = 0.0
    with torch.no_grad():
        for index, (batch_x, batch_y) in enumerate(test_loader):
            pre = net(batch_x.to(device))
            similar = torch.cosine_similarity(batch_y.to(device), pre)
            val_acc += similar.sum().item()
        print("acc", val_acc / test_data_len)
```
**注意:** 这里我把原题的测试集当成验证集来给大家展示了，也就是这里没有测试集。能写验证就能写测试，验证集是在训练里面的，测试集是可以单领出来测试的。  