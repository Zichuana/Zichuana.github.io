---
layout:     post                    # 使用的布局（不需要改）
title:      Metagenomic data analysis code              # 标题 
subtitle:   记录论文中提到的repvgg_b0与csra模块结合方法
date:       2025-1-23              # 时间
author:     zichuana                     # 作者
header-img: img/2025/1.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 生信
---
# Metagenomics Basic Data Processing Code Manual

> [link](https://github.com/Zichuana/Zichuana.github.io/blob/main/doc/%E7%94%9F%E4%BF%A1%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E4%BB%A3%E7%A0%81%E8%AE%B0%E5%BD%95.pdf)

## catalogue  

1. Data Acquisition (prefetch)  

2. Quality Control (fastqc)  

3. Adapter Removal and Contaminant Filtering (cutadapt)  

4. Obtain ASV Feature Table and Taxonomic Annotation Table (dada2)  

5. Genome Assembly (megahit)  

6. Species Composition Analysis  

7. Community α- and β-Diversity Analysis  

8. Species Differential Analysis  

9. Community Differential Analysis  

10. Generate OTU Table (qiime2)  

11. Community Functional Prediction (PICRUSt2)  