---
title: 基于AM-CNN的细菌图谱分类模型
date: 2023-10-06 11:51:47
categories: 
- 课内学习
- 课外项目
tags: 
- 深度学习;
- 自然语言处理;
---

# 1.实现过程

## 1.1模型选择

### 1.1.1基于注意力改进的卷积神经网络算法（AM-CNN）

​	<font color=Red>**AM-CNN**</font>

​	AM-CNN（基于注意力改进的卷积神经网络）模型是一种用于处理细菌拉曼图谱数据的新型深度学习算法。该模型在输入数据特征组合时，考虑了细菌拉曼图谱的波长向量和强度向量，通过滑动窗口方式获取目标词与周围词的综合向量。首先，通过第一次的注意力机制捕获实体与序列中每个词的相关性，并将其与输入的综合词向量矩阵相乘。接着，对卷积结果使用第二次注意力机制捕获视窗与关系的相关性。最终，将卷积结果与相关性矩阵相乘，得到最后的输出结果。

<!--more-->

​	（这个模型的核心在于将细菌拉曼图谱的波长向量和强度向量与输入数据进行组合。首先，将这两种向量进行拼接，构成了最初的输入向量。接着，使用滑动窗口的方式将目标词与周围词组合在一起，形成综合向量。第一次的注意力机制应用在实体与序列中每个词的相关性。将相关性矩阵与输入的综合词向量矩阵相乘，得到一个二维矩阵。然后，使用卷积提取特征，并对卷积结果使用第二次注意力机制捕获视窗与关系的相关性。最后，将卷积结果与相关性矩阵相乘，得到最终的输出结果。通过这种方式，模型能够充分考虑细菌拉曼图谱的波长向量和强度向量在输入数据中的关联关系。）



<img src="image-20230801094141414.png" width="50%" height="50%">

​																			模型结构

网络构建：

1. 输入层：将细菌拉曼图谱的波长向量和强度向量作为输入数据。波长向量和强度向量可以分别作为两个输入通道。
2. 注意力机制1：使用注意力机制1捕获输入数据中实体与序列中每个词的相关性。可以采用自注意力（self-attention）机制或全局平均池化（global average pooling）等方式。
3. 综合词向量矩阵：将注意力机制1得到的相关性矩阵与输入的综合词向量矩阵相乘，得到一个二维矩阵，用于提取特征。
4. 卷积层：使用卷积层对综合词向量矩阵进行特征提取，可以使用不同的卷积核大小和数量，以捕获不同尺度的特征。
5. 注意力机制2：使用注意力机制2对卷积结果进行进一步的特征选择，捕获视窗与关系的相关性。
6. 全连接层：将经过注意力机制2的卷积结果展平，并通过全连接层进行特征融合和映射，得到最终的输出。
7. 输出层：根据任务需求，可以添加合适的输出层，如softmax层用于分类任务，sigmoid层用于二分类任务等。
8. 损失函数：选择合适的损失函数用于模型的训练和优化。

​	

​	我们在训练网络时，为了使得模型可以更快更准确的训练，加入了学习率的自适应调整函数，可以根据训练的数据情况以及已有的训练量来自动调整学习率，使训练效果达到最优。

具体模型构架如下：	

1. 我们首先将训练数据集按照4：1划分成训练集与验证集。
2. 构建AM-CNN网络框架
3. 将训练数据输入AM-CNN网络进行1000轮训练
4. 待模型训练好后，使用测试数据测试模型预测结果
5. 调整模型参数，待模型结构最优后，测试模型最终的分类准确度，并记录训练期间 Loss 值的变动情况。



## 1.2基于注意力改进的卷积神经网络（AM-CNN）实验结果

训练结束后，本实验分别随机选取了3种细菌的50个拉曼数据进行模型评估。

### 1.2.1未标注数据混合

### 1.2.2标注数据混合（6种细菌训练与分类效果）

##### 1.训练准确率变化情况

![训练准确率+评估准确率](训练准确率+评估准确率.png)

准确率变化较为理想，满足预期要求!

- 在第38次训练后模型的**训练准确率**维持在98%
- 在第38次训练后模型的**验证准确率**维持在95%

```
	模型验证通常是在训练过程中使用一个独立于训练集和测试集的数据集进行模型性能评估。它可以用来检测模型是否过拟合或者欠拟合。如果模型在训练集上表现良好，但在验证集上表现较差，那就意味着模型可能过拟合了。这种情况下，可以采取一些方法如提前停止训练或增加正则化等来防止模型过拟合。

	训练准确率代表模型在当前训练数据上的表现。训练多轮后，训练准确率会逐渐提高，这表明模型学到了更多的数据分类特征。但是，如果训练准确率开始变得非常高，而验证准确率却不再提高，这说明模型开始过拟合训练数据。
```

##### 2.训练LOSS值变化情况

![1LOSS](1LOSS.png)

##### 模型分别对于n种细菌数据各自分类情况

![98.833%四种细菌分类情况Process](98.833%四种细菌分类情况Process.png)

![98.75%分类情况process](98.75%分类情况process.png)

**（1）.未标注**

![FREE](FREE.png)

**（2）.标注**

![LABEL](LABEL.png)

##### 5.模型对于测试集的验证情况

![all_bacteria_heatmap](all_bacteria_heatmap.png)

##### 6.模型六种细菌的ROC变化情况

![ROC](ROC.png)

```
	ROC曲线可以帮助我们了解分类器在不同阈值下的表现情况，以及在不同的分类阈值下分类器的敏感性和特异性。曲线的横坐标是假正率（False Positive Rate）即被错误地分为正类的样本占所有负样本的比例，曲线的纵坐标是真正率（True Positive Rate）即被正确地分为正类的样本占所有正样本的比例，曲线越接近左上角，说明分类器的表现越好。    通过ROC曲线我们可以判断分类器的性能是否足够好，同时也可以比较多个分类器的性能，选出最佳的分类器。    举个例子如果ROC曲线下的面积（AUC）接近于1，则说明分类器的性能较好，如果ROC曲线下的面积接近于0.5，则说明分类器的性能不如随机猜测（随机猜测的AUC为0.5）。
```

![PR](C:/Users/m/Desktop/新建文件夹/PR.png)

### 1.2.3与经典网络相比的准确率提升程度

![compare](compare-169657254906614.png)

| Method | 未标注 | 标注  |
| ------ | ------ | ----- |
| ITQ    | 0.615  | 0.628 |
| SH     | 0.684  | 0.744 |
| DSH    | 0.765  | 0.780 |
| SpH    | 0.795  | 0.815 |
| BGAN   | 0.847  | 0.913 |
| AM-CNN | 0.954  | 0.978 |



![performence](performence-169657256481017.png)

## 2.创新点

关于基于细菌拉曼光谱和注意力机制的CNN网络的创新性内容，具有以下几个主要优势：

1.**引入注意力机制**：

传统的卷积神经网络在图像分类任务中，通常使用池化层、全局卷积核等方式提取图像特征。而引入了注意力机制之后，可以使网络更加关注细菌图像的重要特征，从而提高分类精度。

注意力机制在网络中加入一个注意力模块，用于选择和强调输入光谱数据中的重要信息。在该网络中，注意力机制可以结合不同的损失函数进行优化，从而使网络更加有效地学习到重要特征，提高分类的效果。

2.**应用自适应阈值策略**：

传统的细菌分类算法基于训练集的特征设定分类阈值，在测试集上运用时分类效果可能会有所下降。而基于细菌拉曼光谱和注意力机制的CNN网络，可以采用自适应阈值策略，实现对数据特征的自适应调整，避免了传统算法阈值设定不准确的问题。

该网络中引入自适应阈值参数，在网络训练时动态更新自适应阈值**参数**，通过不断的反馈训练数据的特征，不断地调整自适应阈值参数，避免了传统算法阈值设定不准确的问题。

3.**优化模型参数：**

优化模型参数可以提高网络的训练速度和泛化能力，进而提高分类精度。在基于细菌拉曼光谱和注意力机制的CNN网络中，可以通过改变层数、添加跨层连接等方式优化模型的参数，提高分类的效果。

我们的该网络可以通过增加网络层数，引入残差连接、shuffle连接等方式优化模型，增强网络的泛化能力。此外，还可以使用自适应学习率、正则化等技术，进一步优化网络参数。

4.**数据来源新颖**：传统的基于图像的细菌分类方法需要基于显微镜下的图像进行分析和识别。而利用细菌拉曼光谱，则是通过非接触方式直接获取细菌组织的光谱数据，避免了细菌的处理过程对样本造成的影响和污染，同时提供了更加全局、多层次的细菌信息。

5.**数据处理方式创新**：细菌拉曼光谱数据与图像数据不同，需要考虑光谱数据的高维度、数据噪声等问题。基于此，我们运用RamanSpectra库对数据进行预处理和降维，提取有用的特征信息，有利于构建更加高效的分类模型。

#### **部分参考：**

```
1.王吉俐,彭敦陆,陈章,等. AM-CNN:一种基于注意力的卷积神经网络文本分类模型[J]. 小型微型计算机系统,2019,40(4):710-714. DOI:10.3969/j.issn.1000-1220.2019.04.004.

2.Wang, Linlin, et al. "Relation classification via multi-level attention cnns." Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2016.

3.https://mp.weixin.qq.com/s/N-lSzF72TooXAil5FUUW3w
```

## 2.模型部署

```python
import os
import numpy as np
import matplotlib
from keras.models import load_model
from sklearn.model_selection import train_test_split
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from pretty_confusion_matrix import pp_matrix_from_data
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.metrics import precision_recall_curve, average_precision_score
#字体路径
# font = FontProperties(fname='./font/songti.ttf', size=12)
# plt.rcParams['font.sans-serif'] = [font.get_name()]
# plt.rcParams['axes.unicode_minus'] = False

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
matplotlib.rcParams['axes.unicode_minus'] = False
#支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 设置文件夹目录
train_dir = './Final_Data_Ori/train'  # 训练数据文件夹
test_dir = './Final_Data_Ori/test'  # 测试数据文件夹
model_dir = './model'  # 模型保存文件夹
batch_size=1
#自动获取所有的细菌标签
origin_folder_path='./Origin_Data/data2'
labels=[]


# 加载测试数据函数
def load_data(data_dir):
    X = []
    y = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            data = np.loadtxt(file_path)
            if data.ndim < 2:
                data = np.expand_dims(data, axis=0)
            X.append(data)
            label = filename.split("_")[0]
            y.append(label)
    return np.array(X), np.array(y)

# 遍历标签目录
for root, dirs, files in os.walk(origin_folder_path):
    for dir in dirs:
        # 将文件夹名称添加到标签列表中
        labels.append(dir)
labels = [label for label in labels if label != '.ipynb_checkpoints']
print(labels)

X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)
#############################
# 将标签编码为整数
unique_labels = np.unique(y_train)
label_dict = {label: i for i, label in enumerate(unique_labels)}
y_train = np.array([label_dict[label] for label in y_train])
y_test = np.array([label_dict[label] for label in y_test])
num_classes = len(unique_labels)
# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 加载模型85%model.h5
# best_model_path = os.path.join(model_dir, 'best_model.h5')
best_model_path = os.path.join(model_dir, '96%.h5')
model = load_model(best_model_path)

# 进行预测
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

cmap = "PuRd"
pp_matrix_from_data(y_test, y_pred_classes,columns=labels,lw=accuracy,cmap=cmap)
print(y_test)
print('----------------------------------------')
print(y_pred_classes)
print('----------------------------------------')
# print(labels)
print('----------------------------------------')
##########ROC曲线##############
# 画ROC曲线
# 分别绘制每个类别的ROC曲线
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test==i, y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 加颜色和标签
colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of {0} (area = {1:0.2f})'
             ''.format(labels[i], roc_auc[i]))

# 添加一些ROC指令
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()

# 计算每个类别的平均精度分数
average_precision = dict()
for i in range(num_classes):
    average_precision[i] = average_precision_score(y_test == i, y_pred[:, i])

# 计算每个类别的精度-召回率曲线
precision = dict()
recall = dict()
for i in range(num_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test == i, y_pred[:, i])

# 绘制每个类别的精度-召回率曲线
colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
for i, color in zip(range(num_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label='Precision-Recall curve of {0} (area = {1:0.2f})'
             ''.format(labels[i], average_precision[i]))

# 添加一些PR曲线指令
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(loc="lower right")
plt.show()
```

