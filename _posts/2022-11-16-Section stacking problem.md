---
layout:     post                    # 使用的布局（不需要改）
title:      三维胸部CT影像展示案例              # 标题 
subtitle:   从nii文件中获取单向切片，再堆叠成三维图像存在的问题  #副标题
date:       2022-11-16              # 时间
author:     zichuana                     # 作者
header-img: img/2022-11-16/page.jpg   #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - itk/vtk
---

>使用SimpleITK与vtk组件对胸部三维影像进行处理

itk-SNAP展示[数据集](https://www.kaggle.com/code/andrewmvd/covid-19-ct-scans-getting-started/data?select=lung_and_infection_mask)中`Lung and infection mask folder`已分割的nii影像  
![image](/img/2022-11-16/a.png)  
与之相对应的未分割图像  
![image](/img/2022-11-16/b.png)  
### 使用SimpleITK组件读取影像，vtk组件进行图像展示  
```python
from vtk.util.vtkImageImportFromArray import *
import vtk
import SimpleITK as sitk
import numpy as np
import time

# path = '../vtk/nii_data_low/1_1.nii' #segmentation volume
# path = "C:/Users/Zichuana/Desktop/archive (2)/ct_scans/coronacases_002.nii"  # segmentation volume
path = "C:/Users/Zichuana/Desktop/coronacases_002.nii"
# path = "./text.nii"
ds = sitk.ReadImage(path)  # 读取nii数据的第一个函数sitk.ReadImage
print('ds: ', ds)
data = sitk.GetArrayFromImage(ds)  # 把itk.image转为array
print('data: ', data)
print('shape_of_data', data.shape)

# 去掉Hu值小于x的点
# time_start=time.time()
# sum = 0
# for i,iindex in enumerate(data):
#     for j,jindex in enumerate(data[i]):
#         for k,kindex in enumerate(data[j]):
#             sum = sum + 1
#             if data[i][j][k] < 1129:
#                 data[i][j][k] = -1024
# time_end=time.time()
# print('time cost',time_end-time_start,'s')
# sum

spacing = ds.GetSpacing()  # 三维数据的间隔
print('spacing_of_data', spacing)
# data = data[50:]
# data = data[:,:,300:]
srange = [np.min(data), np.max(data)]
print('shape_of_data_chenged', data.shape)
img_arr = vtkImageImportFromArray()  # 创建一个空的vtk类-----vtkImageImportFromArray
print('img_arr: ', img_arr)
print('data:\n', data)
img_arr.SetArray(data)  # 把array_data塞到vtkImageImportFromArray（array_data）
img_arr.SetDataSpacing(spacing)  # 设置spacing
origin = (0, 0, 0)
img_arr.SetDataOrigin(origin)  # 设置vtk数据的坐标系原点
img_arr.Update()
# srange = img_arr.GetOutput().GetScalarRange()

print('spacing: ', spacing)
print('srange: ', srange)


# 键盘控制交互式操作
class KeyPressInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, parent=None):
        self.parent = vtk.vtkRenderWindowInteractor()
        if (parent is not None):
            self.parent = parent

        self.AddObserver("KeyPressEvent", self.keyPress)

    def keyPress(self, obj, event):
        key = self.parent.GetKeySym()
        if key == 'Up':
            gradtfun.AddPoint(-100, 1.0)
            gradtfun.AddPoint(10, 1.0)
            gradtfun.AddPoint(20, 1.0)

            volumeProperty.SetGradientOpacity(gradtfun)
            # 下面这一行是关键，实现了actor的更新
            renWin.Render()
        if key == 'Down':
            tfun.AddPoint(1129, 0)
            tfun.AddPoint(1300.0, 0.1)
            tfun.AddPoint(1600.0, 0.2)
            tfun.AddPoint(2000.0, 0.1)
            tfun.AddPoint(2200.0, 0.1)
            tfun.AddPoint(2500.0, 0.1)
            tfun.AddPoint(2800.0, 0.1)
            tfun.AddPoint(3000.0, 0.1)
            # 下面这一行是关键，实现了actor的更新
            renWin.Render()


def StartInteraction():
    renWin.SetDesiredUpdateRate(10)


def EndInteraction():
    renWin.SetDesiredUpdateRate(0.001)


def ClipVolumeRender(obj):
    obj.GetPlanes(planes)
    volumeMapper.SetClippingPlanes(planes)


ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)  # 把一个空的渲染器添加到一个空的窗口上
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)  # 把上面那个窗口加入交互操作
iren.SetInteractorStyle(KeyPressInteractorStyle(parent=iren))  # 在交互操作里面添加这个自定义的操作例如up,down
min = srange[0]
max = srange[1]
# diff = max - min             #体数据极差
# slope = 4000 / diff
# inter = -slope * min
# shift = inter / slope
# print(min, max, slope, inter, shift)  #这几个数据后面有用
diff = max - min  # 体数据极差
inter = 4200 / diff
shift = -min
print(min, max, inter, shift)  # 这几个数据后面有用

# diffusion = vtk.vtkImageAnisotropicDiffusion3D()
# diffusion.SetInputData(img_arr.GetOutput())
# diffusion.SetNumberOfIterations(10)
# diffusion.SetDiffusionThreshold(5)
# diffusion.Update()

shifter = vtk.vtkImageShiftScale()  # 对偏移和比例参数来对图像数据进行操作 数据转换，之后直接调用shifter
shifter.SetShift(shift)
shifter.SetScale(inter)
shifter.SetOutputScalarTypeToUnsignedShort()
shifter.SetInputData(img_arr.GetOutput())
shifter.ReleaseDataFlagOff()
shifter.Update()

tfun = vtk.vtkPiecewiseFunction()  # 不透明度传输函数---放在tfun
tfun.AddPoint(1129, 0)
tfun.AddPoint(1300.0, 0.1)
tfun.AddPoint(1600.0, 0.12)
tfun.AddPoint(2000.0, 0.13)
tfun.AddPoint(2200.0, 0.14)
tfun.AddPoint(2500.0, 0.16)
tfun.AddPoint(2800.0, 0.17)
tfun.AddPoint(3000.0, 0.18)

gradtfun = vtk.vtkPiecewiseFunction()  # 梯度不透明度函数---放在gradtfun
gradtfun.AddPoint(-1000, 9)
gradtfun.AddPoint(0.5, 9.9)
gradtfun.AddPoint(1, 10)

ctfun = vtk.vtkColorTransferFunction()  # 颜色传输函数---放在ctfun
ctfun.AddRGBPoint(0.0, 0.5, 0.0, 0.0)
ctfun.AddRGBPoint(600.0, 1.0, 0.5, 0.5)
ctfun.AddRGBPoint(1280.0, 0.9, 0.2, 0.3)
ctfun.AddRGBPoint(1960.0, 0.81, 0.27, 0.1)
ctfun.AddRGBPoint(2200.0, 0.9, 0.2, 0.3)
ctfun.AddRGBPoint(2500.0, 1, 0.5, 0.5)
ctfun.AddRGBPoint(3024.0, 0.5, 0.5, 0.5)
# ctfun.AddRGBPoint(0.0, 0.5, 0.0, 0.0)
# ctfun.AddRGBPoint(600.0, 1.0, 255, 0.5)
# ctfun.AddRGBPoint(1280.0, 0.9, 0.2, 255)
# ctfun.AddRGBPoint(1960.0, 255, 0.27, 0.1)
# ctfun.AddRGBPoint(2200.0, 0.9, 0.2, 0.3)
# ctfun.AddRGBPoint(2500.0, 1, 0.5, 0.5)
# ctfun.AddRGBPoint(3024.0, 0.5, 0.5, 0.5)

volumeMapper = vtk.vtkGPUVolumeRayCastMapper()  # 映射器volumnMapper使用vtk的管线投影算法
volumeMapper.SetInputData(shifter.GetOutput())  # 向映射器中输入数据：shifter(预处理之后的数据)
volumeProperty = vtk.vtkVolumeProperty()  # 创建vtk属性存放器,向属性存放器中存放颜色和透明度
volumeProperty.SetColor(ctfun)
volumeProperty.SetScalarOpacity(tfun)
# volumeProperty.SetGradientOpacity(gradtfun)
volumeProperty.SetInterpolationTypeToLinear()  # ???
volumeProperty.ShadeOn()

newvol = vtk.vtkVolume()  # 演员
newvol.SetMapper(volumeMapper)
newvol.SetProperty(volumeProperty)

outline = vtk.vtkOutlineFilter()
outline.SetInputConnection(shifter.GetOutputPort())

outlineMapper = vtk.vtkPolyDataMapper()
outlineMapper.SetInputConnection(outline.GetOutputPort())

outlineActor = vtk.vtkActor()
outlineActor.SetMapper(outlineMapper)

ren.AddActor(outlineActor)
ren.AddVolume(newvol)
ren.SetBackground(0, 0, 0)
renWin.SetSize(600, 600)

planes = vtk.vtkPlanes()

boxWidget = vtk.vtkBoxWidget()
boxWidget.SetInteractor(iren)
boxWidget.SetPlaceFactor(1.0)
boxWidget.PlaceWidget(0, 0, 0, 0, 0, 0)
boxWidget.InsideOutOn()
boxWidget.AddObserver("StartInteractionEvent", StartInteraction)
boxWidget.AddObserver("InteractionEvent", ClipVolumeRender)
boxWidget.AddObserver("EndInteractionEvent", EndInteraction)

outlineProperty = boxWidget.GetOutlineProperty()
outlineProperty.SetRepresentationToWireframe()
outlineProperty.SetAmbient(1.0)
outlineProperty.SetAmbientColor(1, 1, 1)
outlineProperty.SetLineWidth(9)

selectedOutlineProperty = boxWidget.GetSelectedOutlineProperty()
selectedOutlineProperty.SetRepresentationToWireframe()
selectedOutlineProperty.SetAmbient(1.0)
selectedOutlineProperty.SetAmbientColor(1, 0, 0)
selectedOutlineProperty.SetLineWidth(3)

ren.ResetCamera()
iren.Initialize()
renWin.Render()
iren.Start()

``` 
![image](/img/2022-11-16/c.png)  
### 获取切面组  
```python
import os
import nibabel as nib
import shutil
import imageio

file = 'C:/Users/Zichuana/Desktop/coronacases_002.nii'
img = nib.load(file)
img_fdata = img.get_fdata()
fname = file.replace('.nii', '')
img_f_path = os.path.join(fname)
if not os.path.exists(img_f_path):
    os.mkdir(img_f_path)
(x, y, z) = img.shape
for i in range(z):  # z是图像的序列
    silce = img_fdata[:, :, i]  # 选择哪个方向的切片都可以
    imageio.imwrite(os.path.join(img_f_path, '{}.png'.format(i)), silce)  # 保存图像
```
![image](/img/2022-11-16/d.png)
### 旋转并重新堆叠成nii文件  
```python
import os
import cv2
from PIL import Image
import SimpleITK as sitk
import glob
import numpy as np

dir_path = "C:/Users/Zichuana/Desktop/coronacases_002"
files = os.listdir(dir_path)


def save_array_as_nii_volume(data, filename, reference_name=None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = sitk.GetImageFromArray(data)
    if (reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)


image_arr = glob.glob(str(dir_path) + str("/*"))
label = [i for i in range(len(image_arr))]
allImg = np.zeros([200, 512, 512], dtype='uint8')
for i in label:
    print(i)
    single_image_name = dir_path + '/' + str(i) + '.png'
    img_as_img = Image.open(single_image_name)
    im_rotate = img_as_img.transpose(Image.ROTATE_90)
    # img_as_img.show()
    img_as_np = np.asarray(im_rotate)
    allImg[i, :, :] = img_as_np

# np.transpose(allImg,[2,0,1])
save_array_as_nii_volume(allImg, './res.nii')
```
![image](/img/2022-11-16/e.png)
![image](/img/2022-11-16/f.png)
>处理存在一定的问题

在获取单项切面组后，重新堆叠会呈现处“压缩”的状态。  
在影像相关的多项研究中多采用切面进行处理，若再次基础上仍然以3D呈现还存在一定的问题。
