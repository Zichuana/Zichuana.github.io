---
layout:     post                    # 使用的布局（不需要改）
title:      pFind导出文件数据处理              # 标题 
subtitle:   简单实用的蛋白质组学数据处理脚本      # 副标题
date:       2025-9-6              # 时间
author:     zichuana                     # 作者
header-img: img/2025-9-1/cover.png   #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 生信
    - 蛋白质组学
---
### 提取OX或者OS
```python

import pandas as pd
import requests
import re

files = {
    "Archaea": "annotated_Archaea.xlsx",
    "Bacteria": "annotated_Bacteria.xlsx",
    "Fungi": "annotated_Fungi.xlsx"
}

for group, file in files.items():
    # 1. 读取当前文件
    df = pd.read_excel(file)

    # 2. 从 DE 提取 OX
    def extract_ox(de):
        m = re.search(r'OX=(\d+)', str(de))
        return int(m.group(1)) if m else None

    df['OX'] = df['DE'].apply(extract_ox)

    # 5. 保存新文件
    df.to_excel("annotated_" + group + "_OX.xlsx", index=False)
    print(f"✅ 已生成 annotated_{group}_OX.xlsx")
```

```python
import pandas as pd

# 读取 Excel 文件中的所有工作表
xls = pd.read_excel(r'e:\\Protein\result\\9-4Archiving\\Summry_archaea_bacteria_fungi.xlsx', sheet_name=None)
print(xls)

def extract_species(de):
    if pd.isna(de):
        return None
    if "OS=" in str(de) and "OX=" in str(de):
        return str(de).split("OS=")[1].split(" OX=")[0]
    return None

# 处理每个工作表
for sheet_name, df in xls.items():
    df["Species"] = df["DE"].apply(extract_species)
    output_file = f"annotated_{sheet_name}.xlsx"
    df.to_excel(output_file, index=False)
    print(f"✅ 已生成：{output_file}")

if False:
    # 合并所有工作表为一个总表
    combined = pd.concat(xls.values(), keys=xls.keys())
    combined.reset_index(level=0, inplace=True)
    combined.rename(columns={'level_0': 'Sheet'}, inplace=True)
    combined.to_excel("annotated_combined.xlsx", index=False)
```

### 根据ox从ncbi中获取纲目科属系
```python
import pandas as pd
from ete3 import NCBITaxa

# 读取 Excel 文件
df = pd.read_excel('annotated_Fungi_OX.xlsx', sheet_name='Sheet1')

# 确保 OX 列为整数类型
df['OX'] = pd.to_numeric(df['OX'], errors='coerce').astype('Int64')

# 初始化 NCBI taxonomy
ncbi = NCBITaxa()

# 定义要提取的等级
ranks = ['class', 'genus', 'order', 'family']

# 用于存储 order 和 family 的列表
order_list = []
family_list = []
class_list = []
genus_list = []

for taxid in df['OX'].dropna():
    try:
        lineage = ncbi.get_lineage(taxid)
        names = ncbi.get_taxid_translator(lineage)
        rank_dict = ncbi.get_rank(lineage)
        result = {rank: names[tid] for tid, rank in rank_dict.items() if rank in ranks}
        order_list.append(result.get('order', None))
        family_list.append(result.get('family', None))
        genus_list.append(result.get('genus', None))
        class_list.append(result.get('class', None))
    except Exception as e:
        order_list.append(None)
        family_list.append(None)
        class_list.append(None)
        genus_list.append(None)

# 添加新列到原 DataFrame
df['order'] = order_list
df['family'] = family_list
df['class'] = class_list
df['genus'] = genus_list    

# 覆盖保存回原 Excel 文件
df.to_excel('annotated_Fungi_OX.xlsx', sheet_name='Sheet1', index=False)

print("✅ 完成，已直接在原 Excel 文件中添加列。")
```