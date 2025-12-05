---
layout:     post                    # 使用的布局（不需要改）
title:      CT金属拖影去除              # 标题 
subtitle:   常规方法      # 副标题
date:       2025-12-4              # 时间
author:     zichuana                     # 作者
header-img: img/2025-12-4/a.png   #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 考古
---
```python
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure, restoration
import matplotlib

def show_HU(ct):
    import matplotlib.colors as mcolors

    # ---------- 统一颜色映射 ----------
    vmin, vmax = -1000, 3000          # 与热力图一致
    cmap = plt.cm.jet
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # ---------- 1. 全图 HU 直方图 ----------
    plt.figure(figsize=(6, 3))
    counts_all, bins = np.histogram(ct.ravel(), bins=512, range=(-1000, 3000))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_colors = cmap(norm(bin_centers))
    # 画彩色阶梯
    for bc, col, h in zip(bin_centers, bin_colors, counts_all):
        plt.fill_between([bc-4, bc+4], [0, 0], [h, h],
                         color=col, alpha=0.7, edgecolor='none')
    plt.xlim(-1000, 3000)
    plt.xlabel('HU')
    plt.ylabel('Pixels')
    plt.title('HU histogram (full image)')
    plt.tight_layout()
    # plt.show()

    # ---------- 2. 数量<1000 的 HU 直方图（同一颜色） ----------
    roi = ct[:1000, :]                       # 纵坐标 0~999 的区域
    counts_roi, _ = np.histogram(roi.ravel(), bins=bins)  # 复用 bins
    counts_roi = np.clip(counts_roi, 0, 999)               # 只留数量<1000

    plt.figure(figsize=(6, 3))
    for bc, col, h in zip(bin_centers, bin_colors, counts_roi):
        plt.fill_between([bc-4, bc+4], [0, 0], [h, h],
                         color=col, alpha=0.7, edgecolor='none')
    plt.xlim(-1000, 3000)
    plt.xlabel('HU')
    plt.ylabel('Pixels (<1000)')
    plt.title('HU histogram (count < 1000)')
    plt.tight_layout()
    # plt.show()          # 你原来注释掉就保持注释

    # ---------- 3. HU 彩色热力图 ----------
    plt.figure(figsize=(6, 6))
    im = plt.imshow(ct, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04, label='HU')
    plt.title('HU heat-map')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close('all')
    # exit()

# 1. 读取DICOM并转换为HU单位
def read_dicom_hu(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    hu = img * ds.RescaleSlope + ds.RescaleIntercept
    return hu

# 2. 提取金属区域（HU > 2000）
def extract_metal_mask(ct_hu, threshold=2200):
    mask = (ct_hu > threshold).astype(np.uint8)
    return mask

# 4. 图像域
def inpaint_opencv(ct_hu, mask):
    # 归一化到0-255用于inpaint
    img = exposure.rescale_intensity(ct_hu, out_range=(0, 255)).astype(np.uint8)
    mask = mask.astype(np.uint8) * 255
    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return inpainted

# 经典非线性降噪
def median_filter(ct_hu):
    img = exposure.rescale_intensity(ct_hu, out_range=(0, 255)).astype(np.uint8)
    return cv2.medianBlur(img, 5)

# 全变分（Total-Variation）去噪
def denoise_tv(ct_hu):
    img = exposure.rescale_intensity(ct_hu, out_range=(0, 1))
    denoised = restoration.denoise_tv_chambolle(img, weight=0.1)
    return exposure.rescale_intensity(denoised, out_range=(0, 255))

def denoise_tv_th(ct_hu, hu_min=-1500, hu_max=200, weight=0.1):
    """
    mask 内：TV 去噪（其余涂黑） → 再把原图非 mask 区域拼回去
    返回 0-255 uint8 图像
    """
    import numpy as np
    from skimage import exposure, restoration

    # 1. 窗内 mask
    mask = (ct_hu >= hu_min) & (ct_hu <= hu_max)

    # 2. 全局归一化 → TV
    img = exposure.rescale_intensity(ct_hu, out_range=(0, 1))

    # 3. 先全黑，再填入 TV 结果（仅 mask 区域）
    tv_mask = np.zeros_like(img)
    tv_mask[mask] = img[mask]
    tv_black = restoration.denoise_tv_chambolle(tv_mask, weight=weight)
    plt.figure(figsize=(6, 6))
    plt.imshow(tv_mask, cmap='gray')
    plt.title(f"TV mask [{hu_min}, {hu_max}]")
    plt.axis('off')
    plt.tight_layout()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(tv_black, cmap='gray')
    plt.title(f"TV Denoise Th [{hu_min}, {hu_max}]")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close('all')

    # 4. 把原图非 mask 区域贴回来
    hu_range = ct_hu.max() - ct_hu.min()
    tv_hu = tv_black * hu_range + ct_hu.min()          # 转回 HU 空间
    out_hu = ct_hu.copy()
    out_hu[mask]  = tv_hu[mask]   # mask 内用 TV
    # 非 mask 区域保持 ct_hu 原值（已满足）
    # 还原到 [0, 255]
    return exposure.rescale_intensity(out_hu, out_range=(0, 255)).astype(np.uint8)

def inpaint_denoise_tv_th(ct_hu, hu_min=-1500, hu_max=200, weight=0.1, threshold=2200):
    mask_in = extract_metal_mask(ct_hu, threshold=threshold)
    inpainted = inpaint_opencv(ct_hu, mask_in)
    plt.figure(figsize=(6, 6))
    plt.imshow(inpainted, cmap='gray')
    plt.title(f"inpainted [{threshold}]")
    plt.axis('off')
    plt.tight_layout()
    mask = (ct_hu >= hu_min) & (ct_hu <= hu_max)

    img = exposure.rescale_intensity(ct_hu, out_range=(0, 1))
    tv_mask_all = np.zeros_like(img)
    tv_mask_all[mask] = img[mask]
    tv_black_all = restoration.denoise_tv_chambolle(tv_mask_all, weight=weight)

    tv_mask_imp = np.zeros_like(inpainted)
    tv_mask_imp[mask] = inpainted[mask]
    tv_black_imp = restoration.denoise_tv_chambolle(tv_mask_imp, weight=weight)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 两行三列
    axes = axes.ravel()
    axes[0].imshow(tv_black_all, cmap='gray')
    axes[0].set_title(f"TV Denoise Th ALL [{hu_min}, {hu_max}]", fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(tv_mask_all, cmap='gray')
    axes[1].set_title(f"TV Denoise Th ALL MASK", fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(tv_black_imp, cmap='gray')
    axes[2].set_title(f"TV Denoise Th INP [{hu_min}, {hu_max}]", fontsize=10)
    axes[2].axis('off')

    axes[3].imshow(tv_mask_imp, cmap='gray')
    axes[3].set_title(f"TV Denoise Th INP MASK", fontsize=10)
    axes[3].axis('off')
    plt.tight_layout()  
    plt.show()
    plt.close('all')
    # exit()

    # 4. 把原图非 mask 区域贴回来
    # 转回 HU 空间
    hu_range = ct_hu.max() - ct_hu.min()
    inpainted_hu = inpainted * hu_range + ct_hu.min()          # 转回 HU 空间
    inpainted_hu_range = inpainted_hu.max() - inpainted_hu.min()
    tv_hu = tv_black_imp * inpainted_hu_range + inpainted_hu.min()          # 转回 HU 空间
    out_hu = inpainted_hu.copy()
    out_hu[mask]  = tv_hu[mask]    # mask 内用 TV
    return exposure.rescale_intensity(out_hu, out_range=(0, 255)).astype(np.uint8)

# 5. 汇总图
def create_summary(ct_hu, artifact, results, file_name):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 两行三列
    axes = axes.ravel()

    def show(i, img, title):
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')

    # 0 号位：原图
    show(0, exposure.rescale_intensity(ct_hu, out_range=(0, 255)),
         "Original (HU)")

    # 1 开始放 results
    for idx, (name, img) in enumerate(results, start=1):
        show(idx, img, name)

    # 隐藏剩余空白子图
    for j in range(len(results) + 1, 6):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f"metal_artifact_summary_{file_name}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close('all')

# 6. 主流程
if __name__ == "__main__":
    dir_path = './S2030/'
    file_name = 'I2740'
    file_path = dir_path + file_name
    ct_hu = read_dicom_hu(file_path)  # 无后缀也OK
    threshold=2200
    mask = extract_metal_mask(ct_hu)
    show_HU(ct_hu)
    # artifact = simulate_artifact(ct_hu, mask)

    print("max HU:", ct_hu.max())
    print("金属像素数 > 2000:", mask.sum())

    hu_min=-1500
    hu_max=200
    
    results = [
        ("Inpaint (Telea) [{}]".format(threshold), inpaint_opencv(ct_hu, mask)),
        ("Median Filter", median_filter(ct_hu)),
        ("TV Denoise", denoise_tv(ct_hu)),
        ("TV Denoise Th [{} , {}]".format(hu_min, hu_max), denoise_tv_th(ct_hu, hu_min=hu_min, hu_max=hu_max)),
        ("Inpaint + TV Denoise Th [{} , {}, {}]".format(hu_min, hu_max, threshold), inpaint_denoise_tv_th(ct_hu, hu_min=hu_min, hu_max=hu_max, threshold=threshold)),
    ]

    create_summary(ct_hu, ct_hu, results, file_name)
```
![image](/img/2025-12-4/a.png)  
![image](/img/2025-12-4/b.png)  

Inpaint能去除金属，拖影使用TV Denoise。  
![image](/img/2025-12-4/c.png)  
![image](/img/2025-12-4/d.png)  

其他案例，但需要更具直方图就行调整。  
![image](/img/2025-12-4/t1.png)
![image](/img/2025-12-4/t2.png)
![image](/img/2025-12-4/t3.png)