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

### 通过UniProt ID转kegg
```python
# ========================= 0. 依赖 =========================
import pandas as pd
import requests
from tqdm import tqdm
import re
import time
from io import StringIO

# ========================= 1. 读 Excel =========================
file_name = "Summry_archaea_bacteria_fungi.xlsx"   # 原始文件，包含 3 个 sheet（Archaea / Bacteria / Fungi）

# ========================= 2. 从 AC 列抽 UniProt ID =========================
def extract_uniprot_id(ac_str):
    """
    AC 列典型格式：sp|P12345|PROTEIN_NAME
    我们只要第二段 |P12345| 这一段作为 UniProt AC。
    如果为空或格式不对，返回 None。
    """
    if pd.isna(ac_str):
        return None
    parts = str(ac_str).strip().split('|')
    if len(parts) >= 2:
        return parts[1]
    return None

# ========================= 3. UniProt 官方 ID mapping API =========================
UNIPROT_MAP_URL = "https://rest.uniprot.org/idmapping"   # 2022 新版 REST 端点

def uniprot_map(ids, from_db="UniProtKB_AC-ID", to_db="KEGG", chunk_size=5000):
    """
    把一长串 UniProt AC 分批映射到目标数据库（KEGG/COG/EC）。
    每批 ≤5000 是官方建议上限，最后再拼成一个大 DataFrame。
    """
    # 3.1 分批
    chunks = [ids[i:i+chunk_size] for i in range(0, len(ids), chunk_size)]
    parts = []
    for c in chunks:
        parts.append(_uniprot_map_one_chunk(c, from_db, to_db))
    return pd.concat(parts, ignore_index=True)


def _uniprot_map_one_chunk(ids, from_db, to_db):
    print(f"[{time.strftime('%X')}] 提交 {len(ids)} 条 -> {to_db}")
    r1 = requests.post(
        f"{UNIPROT_MAP_URL}/run",
        data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
        timeout=30
    )
    print(f"[{time.strftime('%X')}] 提交返回码 {r1.status_code}")
    if r1.status_code != 200:
        print(r1.text[:200])
        return pd.DataFrame(columns=["UniprotID", to_db])

    job_id = r1.json().get("jobId")
    print(f"[{time.strftime('%X')}] jobId {job_id}")
    if not job_id:
        return pd.DataFrame(columns=["UniprotID", to_db])

    # --- 轮询上限 120 次（约 3 分钟），防止死循环 ---
    for poll in range(120):
        r2 = requests.get(f"{UNIPROT_MAP_URL}/status/{job_id}", timeout=30)
        print(f"[DEBUG] 轮询 {poll:02d} 返回长度 {len(r2.text)} 字符")

        # 策略 1：出现 results 且长度合理 → 完成
        if r2.status_code == 200 and "results" in r2.text and len(r2.text) > 50:
            print(f"[DEBUG] 检测到 results，job 已完成")
            job_status = "FINISHED"
            break

        # 策略 2：返回长度很小且无 jobStatus → 空结果，也视为完成
        if poll == 0:
            first_len = len(r2.text)
        if len(r2.text) == first_len and poll > 3:
            print(f"[DEBUG] 连续多次相同短响应，视为空结果，跳出")
            job_status = "FINISHED"
            break

        # 策略 3：正常字段判断
        if r2.headers.get("content-type", "").startswith("application/json"):
            job_status = r2.json().get("jobStatus")
            if job_status == "FINISHED":
                break
            if job_status in {"ERROR", "FAILED"}:
                print("[ERROR] Job failed:", r2.text[:200])
                return pd.DataFrame(columns=["UniprotID", to_db])
        time.sleep(1.5)
    else:
        print("[ERROR] 轮询 120 次仍未完成，强制放弃")
        return pd.DataFrame(columns=["UniprotID", to_db])

    # ---------------- 关键：把结果解析成 DataFrame ----------------
    results_url = f"{UNIPROT_MAP_URL}/results/{job_id}"
    r3 = requests.get(results_url, timeout=30)
    if r3.status_code != 200:
        print("[ERROR] 获取 results 失败:", r3.text[:200])
        return pd.DataFrame(columns=["UniprotID", to_db])

    data = r3.json()
    if not data or "results" not in data or not data["results"]:
        # 空映射
        return pd.DataFrame(columns=["UniprotID", to_db])

    # 整理成两列：from -> to
    records = []
    for item in data["results"]:
        # from 是 UniProtAC，to 是目标库 ID
        records.append({
            "UniprotID": item["from"],
            to_db: item["to"]
        })
    return pd.DataFrame(records)
    
# ========================= 4. 处理单个 kingdom =========================
def process_kingdom(sheet_name):
    """
    对单个 sheet（Archaea/Bacteria/Fungi）：
    1) 读表 → 抽 UniProt ID
    2) 分别映射 KEGG(KO)、COG、EC 号
    3) 合并回原表注释列
    4) 输出 csv
    """
    print(f"\n📊 处理 {sheet_name}...")
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    df['UniprotID'] = df['AC'].apply(extract_uniprot_id)
    ids = df['UniprotID'].dropna().unique().tolist()
    print(f"共 {len(ids)} 个 UniProt ID")

    # 4.1 三次映射
    ko  = uniprot_map(ids, "UniProtKB_AC-ID", "KEGG")   # KO 号
    # cog = uniprot_map(ids, "UniProtKB_AC-ID", "COG")    # COG 类
    # ec  = uniprot_map(ids, "UniProtKB_AC-ID", "Enzyme") # EC 号

    # 4.2 把注释列（DE、PSM Count…）先去重，再跟三张映射表 left join
    annot = df[['UniprotID', 'DE', 'PSM Count', 'Coverage', 'Score']].drop_duplicates(subset=['UniprotID'])
    annot = annot.merge(ko,  on='UniprotID', how='left')
    # annot = annot.merge(cog, on='UniprotID', how='left')
    # annot = annot.merge(ec,  on='UniprotID', how='left')

    # 4.3 保存
    output = f"{sheet_name.lower()}_annot.csv"
    annot.to_csv(output, index=False)
    print(f"✅ 已保存：{output}")
    return annot

# ========================= 5. 主流程 =========================
for kingdom in ["Bacteria", "Archaea", "Fungi"]:
    process_kingdom(kingdom)
```

### 结论
针对土壤蛋白质组学不适合做功能富集等下游分析，结合基因组学可能会有意义，通过 https://www.uniprot.org/id-mapping/ 转换成eggnog-mapper在其他数据中可能会有意义，https://zhuanlan.zhihu.com/p/113015245