---
layout:     post                    # 使用的布局（不需要改）
title:      创建python库方法指南               # 标题 
subtitle:   一个笔记嗯
date:       2023-3-3              # 时间
author:     zichuana                     # 作者
header-img: img/post-bg-desk2.png    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 碎碎念
---
### 创建
首先确定所创建的python库没有重名，假定为XXX。  
本地新建一个文件夹名字XXX。  
XXX文件夹格式如下：  
```
XXX
├─XXX
│  ├─__init__.py
│  └─XXX.py
├─LICENSE
├─README.md
└─setup.py   
```
**__init__.py**  
初始化启动函数  
``` python
from __future__ import absolute_import
from .XXX import *
name = "XXX"
```

**XXX.py**  
XXX的主函数，先随便写一个def()，用于调试。  
```python
# XXX Main
import pandas as pd
import numpy as np

def info():
    print('wzh is pig')
```
**LICENSE**  
借助MIT License模板  
```
Copyright (C) 2020 The Python Package Authority

Permission is hereby granted, free of charge, to any person obtaining a copy of this Software and associated documentations files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
**README.md**  
用于发布到PyPI平台后的项目介绍内容展示  
```
This is a test.
```
**setup.py**  
``` python
import setuptools

with open("README.md",'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name = "XXX",
    version = "0.0.1",
    author = "XXX",
    author_email = "@.com",
    description = "This is a test.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url="https://github.com/",
    packages=setuptools.find_packages(),
    install_requires=['pandas','matplotlib','numpy','scipy','pandas_profiling','folium','seaborn'],
    # add any additional packages that needs to be installed along with SSAP package.

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
```
### 发布  
在XXX下运行  
`python setup.py sdist bdist_wheel`  
保证含有`twine`  
`pip install twine`  
发布(需要输入账号密码)  
`twine upload dist/*`  
### 使用
`pip install TAILab`  
打开python  
```
import XXX
XXX.info()
```
### 更新
```
python setup.py sdist bdist_wheel
twine upload dist/*
```