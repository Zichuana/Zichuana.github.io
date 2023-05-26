---
layout:     post                    # ʹ�õĲ��֣�����Ҫ�ģ�
title:      ���ѧϰ�ڿμ�¼3              # ���� 
subtitle:   2022�˹����ܾ�����������ѧϰѵ�����̽���
date:       2023-5-26              # ʱ��
author:     zichuana                     # ����
header-img: img/2023-5-26/page.jpg    #��ƪ���±��ⱳ��ͼƬ
catalog: true                       # �Ƿ�鵵
tags:                               #��ǩ
    - ���ѧϰ
---
> ���ڽ������ѧϰ����ģ�͵�ѵ������  

22���˹���������ֻҪ�����loss������û�����ѵ��ʱÿ��epoch׼ȷ�ʣ��������Ҫ�аɣ�����ѵ��������֤����acc����Ļ����ǿ��Ժܺõ��ж�ģ���Ƿ���ϣ���������Ҫ����һ�㡣  
��������Ҿ��û��Ǻ��б�Ҫ����һ�£�������Ϊ�������������һЩ������  
ͨ��ѵ��һ��ģ������Ҫ���ֳɣ�ѵ��������֤���Ͳ��Լ��ġ��ڲ��Լ�����֤����ѵ�����ڲ��Լ��ϲ��ԣ����߶�������Ϊ����ģ�͵ı�׼��  
������һ��Ľ��ܣ��Ҿ���дע�������ܴ��룬ԭ�����Ȥ����������ʦ�ɣ�  
```python
train_steps = len(train_loader)  # ��ȡÿһ��epoch�ڴ���ͼ���������һ��16��
net = NET(mask_len)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # ����һ���Ż�����������ѡ����ģ��ѵ���׶���ʹ�õ��Ż���
loss_func = torch.nn.L1Loss()  # ����һ����ʧ����
for epoch in range(10):   # ����epoch 10����˼һ��
    train_loss = 0.0  # ָ����ÿһ��epoch��loss��acc ��Ҫ������index������
    train_acc = 0.0
    for index, (batch_x, batch_y) in enumerate(train_loader):  # �������train_loader��ͼ���ͽ�ȥ
        out = net(batch_x.to(device))  # ������������ѧϰ�Ľ��
        loss = loss_func(out, batch_y)  # ʹ����ʧ����������һ��index����ʧֵ
        optimizer.zero_grad()  # �Ż�������
        loss.backward()  # ��ʧ������򴫲�
        optimizer.step()  # �Ż���ʹ��
        train_loss += loss.item()  # �ӵ�train_loss���棬����train_loss���Դ���ͼ�������������һ��epoch��lossƽ��ֵ�ˣ������ֵ
        similar = torch.cosine_similarity(out, batch_y.to(device))  # �������ƶ�
        train_acc += similar.sum().item()  # �����ƶ����ϵ���׼ȷ���У���������ܵ�ͼ���������õ�ÿ��ͼ���ƽ��׼ȷ��
    print('Epoch:{} Train Loss:{} acc:{}'.format(epoch + 1, train_loss / train_steps, train_acc / train_data_len))

    net.eval()  # ���ڷ��򴫲��ˣ������ǿ�ʼ���Ի�����֤�ı�־
    val_acc = 0.0
    with torch.no_grad():
        for index, (batch_x, batch_y) in enumerate(test_loader):
            pre = net(batch_x.to(device))
            similar = torch.cosine_similarity(batch_y.to(device), pre)
            val_acc += similar.sum().item()
        print("acc", val_acc / test_data_len)
```
**ע��:** �����Ұ�ԭ��Ĳ��Լ�������֤���������չʾ�ˣ�Ҳ��������û�в��Լ�����д��֤����д���ԣ���֤������ѵ������ģ����Լ��ǿ��Ե���������Եġ�  