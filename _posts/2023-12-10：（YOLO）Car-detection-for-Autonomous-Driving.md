---
title: 2023-12-10：（YOLO）Car detection for Autonomous Driving
copyright: true
mathjax: true
date: 2023-12-10 10:53:59
categories:
- 课内学习
- 传统算法学习
- 吴恩达课程学习
- Convolutional Neural Networks
tags:
- YOLO;
- 目标检测;
---

## 1.任务

​	构建YOLO算法的各个模块：Bounding Box预测、交并比（IoU）、非极大值抑制、Anchor Boxes。

​	然后利用构建的YOLO算法进行目标识别（汽车检测）。

<!-- more -->

## 2.导入依赖包

```python
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
```

## 3.YOLO

### 3.1Model Details

首先我们知道：

- **输入**是一批形状为 (m, 608, 608, 3) 的图像
- **输出**是一系列边界框以及识别出的类别。如上所述，每个边界框由6个数字 \$(p_c, b_x, b_y, b_h, b_w, c)\$ 表示。如果将 \$c\$ 扩展为一个80维向量，那么每个边界框将由85个数字表示。

我们将使用5个锚框。因此，你可以将YOLO架构想象为以下流程：图像 (m, 608, 608, 3) -> 深度CNN -> 编码 (m, 19, 19, 5, 85)。

让我们更详细地了解这种编码代表什么。

<img src="https://s2.loli.net/2023/12/10/dcvZI2g18GRPzoV.png" alt="image-20231210111606861" style="zoom:91%;" />

如果一个物体的中心/中点落在一个网格单元中，那么该网格单元负责检测该物体。

由于我们使用了5个锚框，因此每个19x19的单元格都包含了5个框的信息。锚框仅由其宽度和高度定义。

为了简化，我们将展平形状的最后两个维度 (19, 19, 5, 85) 编码。所以深度CNN的输出是 (19, 19, 425)。

![image-20231210111623155](https://s2.loli.net/2023/12/10/jTe5BVqnYZrgiu2.png)

现在，对于每个单元格的每个框，我们将计算以下逐元素乘积，并提取框包含某个类别的概率。

![image-20231210111647569](https://s2.loli.net/2023/12/10/mrOgZadx6NeoHuP.png)

以下是一种可视化YOLO在图像上预测的方式：
- 对于19x19网格的每个单元格，找出概率分数的最大值（在5个锚框和不同类别中取最大值）。
- 根据该网格单元格最可能考虑的对象对该网格单元进行着色。

这样做的结果是这样的图片：

![image-20231210111700040](https://s2.loli.net/2023/12/10/IfszkTFgvCKABWc.png)

请注意，这种可视化并不是YOLO算法本身用于进行预测的核心部分；它只是一种可视化算法中间结果的好方法。
另一种可视化YOLO输出的方式是绘制它输出的边界框。这样做的结果是这样的可视化：

![image-20231210111706691](https://s2.loli.net/2023/12/10/Ot5nSRLF7fHbYvE.png)

在上图中，我们只绘制了模型分配了高概率的框，但这仍然是太多的框。你希望将算法的输出过滤到更少的检测到的物体。为此，你将使用非最大抑制。具体来说，你将执行以下步骤：
- 去除分数低的框（意味着，该框对检测类别的置信度不高）
- 当几个框重叠在一起并检测到同一个物体时，只选择一个框。

### 3.2使用类别得分阈值进行过滤

你将通过设置阈值来应用第一个过滤器。你希望去除那些类别“得分”低于选定阈值的框。

模型给出了总共19x19x5x85个数字，每个框由85个数字描述。将(19,19,5,85)（或(19,19,425)）维度的张量重新排列成以下变量会更方便：
- `box_confidence`：形状为$(19 \times 19, 5, 1)$的张量，包含每个19x19单元格中预测的5个框的$p_c$（存在某物体的置信概率）。
- `boxes`：形状为$(19 \times 19, 5, 4)$的张量，包含每个单元格的5个框的$(b_x, b_y, b_h, b_w)$。
- `box_class_probs`：形状为$(19 \times 19, 5, 80)$的张量，包含每个单元格的5个框的80个类别的检测概率$(c_1, c_2, ... c_{80})$。

**练习**：实现`yolo_filter_boxes()`。

**代码：**

```python
# 评分函数：yolo_filter_boxes（过滤YOLO边框）

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """
    通过阈值处理来过滤YOLO的边框，基于物体和类别的置信度。

    参数:
    box_confidence -- 形状为 (19, 19, 5, 1) 的张量，表示边框的置信度
    boxes -- 形状为 (19, 19, 5, 4) 的张量，表示边框的坐标
    box_class_probs -- 形状为 (19, 19, 5, 80) 的张量，表示各个类别的概率
    threshold -- 实数值，如果[最高类别概率得分 < 阈值]，则丢弃对应的边框
    
    返回值:
    scores -- 形状为 (None,) 的张量，包含选中边框的类别概率得分
    boxes -- 形状为 (None, 4) 的张量，包含选中边框的(b_x, b_y, b_h, b_w)坐标
    classes -- 形状为 (None,) 的张量，包含被选中边框检测到的类别索引
    
    注意: "None"是因为你不知道选中的边框的确切数量，这取决于阈值。
    例如，如果有10个边框，scores的实际输出大小将是(10,)。
    """
    
    # 第1步：计算边框得分
    ### 开始编写代码 ### (≈ 1 行)
    box_scores = box_confidence * box_class_probs
    ### 结束编写代码 ###
    
    # 第2步：根据最大的box_scores找到box_classes，并跟踪相应的得分
    ### 开始编写代码 ### (≈ 2 行)
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1, keepdims=False)
    ### 结束编写代码 ###
    
    # 第3步：基于 "box_class_scores" 使用 "threshold" 创建一个过滤掩码。掩码应该与box_class_scores的维度相同，
    # 并且对于你想要保留的边框（概率 >= 阈值）应为True
    ### 开始编写代码 ### (≈ 1 行)
    filtering_mask = box_class_scores > threshold
    ### 结束编写代码 ###
    
    # 第4步：对scores、boxes和classes应用掩码
    ### 开始编写代码 ### (≈ 3 行)
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    ### 结束编写代码 ###
    
    return scores, boxes, classes
```

**部署：**

```python
with tf.Session() as test_a:
    box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
    boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
    box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.shape))
    print("boxes.shape = " + str(boxes.shape))
    print("classes.shape = " + str(classes.shape))
```

**输出：**

```python
scores[2] = 10.750582
boxes[2] = [ 8.426533   3.2713668 -0.5313436 -4.9413733]
classes[2] = 7
scores.shape = (?,)
boxes.shape = (?, 4)
classes.shape = (?,)
```

### 3.3非极大值抑制

​	即使通过对类别分数进行阈值过滤，最终仍会出现大量重叠的方框。第二种筛选方法是非极大值抑制（NMS）。

![image-20231210112917826](https://s2.loli.net/2023/12/10/H3aph8TvAgMXRJN.png)

​	在上图中，模型预测了 3 辆车，但实际上是对同一辆车的 3 次预测。运行非最大抑制 (NMS) 将只选择 3 个方框中最准确（概率最高）的一个。

非极大值抑制使用一个非常重要的函数，称为**“交集与并集”**，或者IoU。
<img src="https://s2.loli.net/2023/12/10/Qa6y2r4olkhszED.png" alt="iou" style="zoom: 50%;" />

**练习**：实现iou()函数。一些提示：
- 在这个练习中，我们使用两个角点（左上和右下）来定义一个盒子：(x1, y1, x2, y2)，而不是中点和高度/宽度。
- 要计算矩形的面积，你需要将其高度（y2 - y1）乘以宽度（x2 - x1）
- 你还需要找到两个盒子相交的坐标(xi1, yi1, xi2, yi2)。记住：
    - xi1 = 两个盒子的x1坐标的最大值
    - yi1 = 两个盒子的y1坐标的最大值
    - xi2 = 两个盒子的x2坐标的最小值
    - yi2 = 两个盒子的y2坐标的最小值
    

在这段代码中，我们采用的惯例是(0,0)是图像的左上角，(1,0)是右上角，(1,1)是右下角。

**代码：**

```python
# 评分函数：iou（计算交并比）

def iou(box1, box2):
    """
    实现 box1 和 box2 之间的交并比（IoU）。

    参数:
    box1 -- 第一个边框，列表对象，坐标为 (x1, y1, x2, y2)
    box2 -- 第二个边框，列表对象，坐标为 (x1, y1, x2, y2)
    """

    # 计算 box1 和 box2 交集的 (y1, x1, y2, x2) 坐标，并计算其面积。
    ### 开始编写代码 ### (≈ 5 行)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    ### 结束编写代码 ###    

    # 通过公式 Union(A,B) = A + B - Inter(A,B) 计算并集面积
    ### 开始编写代码 ### (≈ 3 行)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    ### 结束编写代码 ###
    
    # 计算 IoU
    ### 开始编写代码 ### (≈ 1 行)
    iou = inter_area / union_area
    ### 结束编写代码 ###

    return iou
```

**部署：**

```python
box1 = (2, 1, 4, 3)
box2 = (1, 2, 3, 4) 
print("iou = " + str(iou(box1, box2)))
```

**输出：**

```python
iou = 0.14285714285714285
```

​	现在已经准备好实现非极大值抑制的前提条件了。

关键步骤如下： 

- 选择得分最高的盒子。
- 计算它与所有其他盒子的重叠部分，并移除那些与它重叠超过`iou_threshold`的盒子。
- 返回第1步并迭代，直到没有更低分数的盒子。 这将移除所有与选定盒子重叠较大的盒子。只留下“最佳”盒子。

**代码：**

```python
# 评分函数：yolo_non_max_suppression（实施非极大值抑制）

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    对一组边框应用非极大值抑制（NMS）。

    参数:
    scores -- 形状为 (None,) 的张量，yolo_filter_boxes() 的输出
    boxes -- 形状为 (None, 4) 的张量，yolo_filter_boxes() 的输出，已缩放到图像大小（见后文）
    classes -- 形状为 (None,) 的张量，yolo_filter_boxes() 的输出
    max_boxes -- 整数，你希望预测的最大边框数量
    iou_threshold -- 实数值，用于NMS过滤的“交并比”阈值
    
    返回值:
    scores -- 形状为 (, None) 的张量，每个边框的预测得分
    boxes -- 形状为 (4, None) 的张量，预测的边框坐标
    classes -- 形状为 (, None) 的张量，每个边框的预测类别
    
    注意：输出张量的 "None" 维度显然必须小于 max_boxes。还要注意，这个函数会转置 scores, boxes, classes 的形状。这是为了方便处理。
    """
    
    # 使用 tf.image.non_max_suppression() 获取对应于你保留的边框的索引列表
    ### 开始编写代码 ### (≈ 1 行)
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold, name=None)
    ### 结束编写代码 ###
    
    # 使用 K.gather() 从 scores, boxes 和 classes 中只选择 nms_indices
    ### 开始编写代码 ### (≈ 3 行)
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    ### 结束编写代码 ###
    
    return scores, boxes, classes
```

**部署：**

```python
with tf.Session() as test_b:
    scores = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 1)
    classes = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))
```

**输出：**

```python
scores[2] = 6.938395
boxes[2] = [-5.299932    3.1379814   4.450367    0.95942086]
classes[2] = -2.2452729
scores.shape = (10,)
boxes.shape = (10, 4)
classes.shape = (10,)
```

### 3.4结束过滤

​	利用深度 CNN 的输出（19x19x5x85 维编码）实现一个函数，并使用刚才实现的函数对所有方框进行过滤。

**练习**： 

实现 **yolo_eval()**，获取 YOLO 编码的输出，并使用分数阈值和 NMS 过滤方框。还有最后一个实现细节你必须知道。有几种表示方框的方法，例如通过角或通过中点和高/宽。YOLO 会使用以下函数（依赖包提供了这些函数）在不同时间转换几种此类格式： 

```python
boxes = yoloo_boxes_too_corners(box_xy, box_wh) 
```
将 yolo 方框坐标（x,y,w,h）转换为方框角坐标（x1, y1, x2, y2），以适应 `yoloo_filter_boxes` 的输入。
```python
boxes = scale_boxes(boxes, image_shape)
```
YOLO 的网络是在 608x608 的图像上训练运行的。如果您要在不同尺寸的图像上测试这些数据，例如，汽车检测数据集的图像尺寸为 720x1280，那么这一步将重新缩放方框，以便将它们绘制在原始的 720x1280 图像之上。 

**代码：**

```python
# 评分函数：yolo_eval（YOLO评估）

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    将YOLO编码的输出（大量边框）转换为预测的边框及其分数、边框坐标和类别。

    参数:
    yolo_outputs -- 编码模型的输出（对于 (608, 608, 3) 的图像形状），包含4个张量：
                    box_confidence: 形状为 (None, 19, 19, 5, 1) 的张量
                    box_xy: 形状为 (None, 19, 19, 5, 2) 的张量
                    box_wh: 形状为 (None, 19, 19, 5, 2) 的张量
                    box_class_probs: 形状为 (None, 19, 19, 5, 80) 的张量
    image_shape -- 形状为 (2,) 的张量，包含输入形状，在本笔记本中我们使用 (608., 608.)（必须是 float32 数据类型）
    max_boxes -- 整数，你希望预测的最大边框数量
    score_threshold -- 实数值，如果[最高类别概率得分 < 阈值]，则丢弃对应的边框
    iou_threshold -- 实数值，用于NMS过滤的“交并比”阈值
    
    返回值:
    scores -- 形状为 (None, ) 的张量，每个边框的预测得分
    boxes -- 形状为 (None, 4) 的张量，预测的边框坐标
    classes -- 形状为 (None,) 的张量，每个边框的预测类别
    """
    
    ### 开始编写代码 ### 
    
    # 获取YOLO模型的输出 (≈1行)
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # 转换边框，准备进行过滤函数处理
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # 使用之前实现的函数进行得分过滤，阈值为 score_threshold (≈1行)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
    
    # 将边框缩放回原始图像形状
    boxes = scale_boxes(boxes, image_shape)

    # 使用之前实现的函数进行非极大值抑制，阈值为 iou_threshold (≈1行)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
    
    ### 结束编写代码 ###
    
    return scores, boxes, classes
```

**部署：**

```python
with tf.Session() as test_b:
    yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
    scores, boxes, classes = yolo_eval(yolo_outputs)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))
```

**输出：**

```python
scores[2] = 138.79124
boxes[2] = [1292.3297  -278.52167 3876.9893  -835.56494]
classes[2] = 54
scores.shape = (10,)
boxes.shape = (10, 4)
classes.shape = (10,)
```

