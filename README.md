# 百度网盘AI大赛——图像处理挑战赛：文档图像摩尔纹消除第1名方案

## 1.竞赛分析

比赛链接：[百度网盘AI大赛：文档图像摩尔纹消除(赛题一)](https://aistudio.baidu.com/aistudio/competition/detail/128/0/task-definition)

本竞赛的任务是利用AI算法消除图像中的摩尔纹，其实质还是image2image的low level任务。


## 2.数据处理

组委会给定的训练集只有1000对moire-sharp图像，同时由于图像的分辨率非常大且不相同，没办法直接训练，因此我们需要对训练数据进行裁剪和增强。

(1) 以60%的重叠率将图像切分成512 x 512的patch；

(2) 训练数据增强：水平翻转，竖直翻转。

![](https://ai-studio-static-online.cdn.bcebos.com/8d3369d4cd604e4ca13894612bf62d945c9effb404c7477cbce70654dba61dc5)


## 3.模型结构

### 3.1 算法思路

(1) 放弃基于self-attention的方法

因为算法的推理时间、模型大小和显存大小等也评价指标，因此我们放弃了基于self-attention的方法，包括：Non-Local和Transformer等；

(2) 放弃channel attention (CA)

在RCAN中，作者使用CA模块提升了图像超分辨的效果，但是这种提升在本任务中效果非常小（也许是我们加的方式不对），而且会来额外的参数量，因此我们放弃该方法；

(3) MRNet

我们的MRNet曾获得[NTIRE 2021 Defocus Deblurring using Dual-pixel Images Challenge (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Abuolaim_NTIRE_2021_Challenge_for_Defocus_Deblurring_Using_Dual-Pixel_Images_Methods_CVPRW_2021_paper.pdf)冠军，因此使用MRNet来实现图像去摩尔纹任务。


MRNet是由特征提取模块、融合模块、重建模块和上采样模块构成。

a. 特征提取模块：一个简单的U-Net架构组成，提取输入图像的多尺度特征；

b. 融合模块：一个简单的1x1卷积构成；

c. 重建模块：重建模块是本架构的核心，是由多个Multi-scale Residual Group Module (MSRGM)组成。其中MSRGM融合多尺度特征来提取模型的表达能力，而每个尺度是由多个[Residual Group Module (RGM)](https://arxiv.org/abs/1807.02758)构成；

d. 上采样模块：使用pixel-shuffle模块进行上采样。

![](https://ai-studio-static-online.cdn.bcebos.com/fd375d1ac8bd462c919b7b7e7ff2f5436ea7086f63cc4f128c9508c8e8f058ee)


### 3.2 训练策略

(1) 采用Charbonnier L1 loss function

(2) 采用Cosine学习了调整策略(1000 epoch，lr=1e-4)

(3) input size = 512 * 512


## 4.代码结构

train.py：训练

MRNET.py：模型

losses.py：loss函数

transforms.py：数据增强

predict_demoire_paddle.py：测试

image_to_patch_moire.py：数据预处理