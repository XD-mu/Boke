---
title: Proposal-based-Multiple-Instance-Learning-for-Weakly-supervised-Temporal-Action-Localization（基于提案的弱监督时序动作定位多示例学习）
copyright: true
mathjax: true
date: 2023-12-07 15:32:37
categories:
- 课内学习
- 论文精读
tags:
- 视频定位;
- 弱监督;
- 时序动作定位;
---

## 1.Abstract

​	弱监督时态动作定位的目的是在训练过程中仅使用视频级类别标签来定位和识别未剪辑视频中的动作。在没有实例级注释的情况下，大多数现有方法都遵循基于片段的多实例学习（S-MIL）框架，即通过视频标签对片段预测进行监督。然而，在训练过程中获取分段级分数的目标与在测试过程中获取建议级分数的目标并不一致，从而导致了次优结果。为了解决这个问题，我们提出了一种新颖的基于提案的多实例学习（Proposal-based Multiple Instance Learning，P-MIL）框架，在训练和测试阶段直接对候选提案进行分类，其中包括三个关键设计：

- 周边对比特征提取模块，通过考虑周边对比信息来抑制辨别性短提案；
- 提案完整性评估模块，通过完整性伪标签的指导来抑制低质量提案；
- 实例级等级一致性损失，通过利用 RGB 和 FLOW 模式的互补性来实现鲁棒检测；

​	在两个具有挑战性的基准（包括 THUMOS14 和 ActivityNet）上取得的大量实验结果证明了我们的方法具有卓越的性能。

<!-- more -->

## 2.Motivation

S-MIL 框架有两个缺点。

1. 训练和测试阶段的目标不一致。如图（a）所示，在测试阶段，目标是对动作建议整体进行评分，但在训练阶段，分类器的训练目标是对片段进行评分。
2. 在很多情况下，很难单独对每个片段进行分类。如图（b）所示，通过观察单个跑步片段，很难判断它属于跳高、跳远还是三级跳远。

<img src="https://renhuan1999.github.io/P-MIL/images/P-MIL_motivation.png" alt="img"  />

## 4.Model FrameWork

![img](https://renhuan1999.github.io/P-MIL/images/P-MIL_framework.png)

(a) 基于提案的多实例学习框架概述，包括候选提案生成、提案特征提取、提案分类和完善。

(b) 周边对比特征提取（SCFE）模块扩展候选提案的边界，然后计算候选提案的内外对比特征。

(c) 提案完整性评估（PCE）模块通过计算选定伪实例的 IoU 生成完整性伪标签。

(d) 实例级等级一致性（IRC）损失限制 RGB 和 FLOW 两种模式之间簇内的归一化相对分类分数保持一致。

## 5.Experiment

![img](https://renhuan1999.github.io/P-MIL/images/P-MIL_experiment_thumos14.png)

**Table 1**：这个表格展示了在THUMOS14测试集上，不同方法在mAP（mean Average Precision）指标下的性能对比。这个表格说明了研究者提出的P-MIL方法在弱监督和全监督方法中的表现，结果表明P-MIL方法优于大部分现有方法，甚至在某些指标上超过了一些全监督方法。

![img](https://renhuan1999.github.io/P-MIL/images/P-MIL_experiment_activitynet.png)

**Table 2和Table 3**：这两个表格分别展示了在ActivityNet1.2和ActivityNet1.3验证集上的性能对比。这两个表格进一步证实了P-MIL方法在更大的数据集上也能取得优秀的性能。

![img](https://s2.loli.net/2023/12/08/7Qv9wxEOHPGqfUd.png)

**Table 4**：这个表格展示了不同候选提案生成方法对性能的影响。结果显示，引入背景提案能够显著提升性能，这验证了研究者提出的这一设计的有效性。

**Table 5**：这个表格展示了不同提案评分方法的性能对比，包括直接使用IoU与ground truth评分，使用S-MIL方法评分，使用P-MIL方法评分，以及融合S-MIL和P-MIL方法评分。结果显示，P-MIL方法在评分上优于S-MIL方法，且融合两种方法可以进一步提升性能。

**Table 6**：这个表格展示了不同变体的提案特征提取方法对性能的影响。结果显示，不扩展边界、直接连接三个特征向量、以及使用外-内对比特征提取的方法的性能逐渐提升，验证了环绕对比特征提取模块的有效性。

**Table 7**：这个表格展示了提案完整性评估（PCE）模块和实例级别排名一致性（IRC）损失对性能的影响。结果显示，这两个设计都能显著提升性能，且同时使用可以带来更大的提升。

<img src="https://s2.loli.net/2023/12/08/6Kyh3irNIPBTevo.png" alt="image-20231208101605046" style="zoom:150%;" />

**Figure 3**：这个图展示了两个超参数（λcomp和λIRC）对性能的影响。结果显示，模型对于这些超参数不是非常敏感，性能波动在mAP@0.5中小于2%。

## 6.Visualizations Conclusion

<img src="https://renhuan1999.github.io/P-MIL/images/P-MIL_visualization.png" alt="img"  />

​	本文提出了一种新颖的基于提案的多实例学习（Proposal-based Multiple Instance Learning，**P-MIL**）框架，用于弱监督时空动作定位，通过直接对候选提案进行分类，实现训练和测试阶段的统一目标。作者引入了三个关键设计来应对 P-MIL 框架不同阶段的挑战，包括**周边对比特征提取模块、提案完整性评估模块和实例级等级一致性损失**。在两个具有挑战性的基准上取得的大量实验结果证明了我们方法的有效性。

## 7.相关链接

​	[论文地址](https://openaccess.thecvf.com/content/CVPR2023/papers/Ren_Proposal-Based_Multiple_Instance_Learning_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.pdf)

​	[作者源码地址](https://github.com/RenHuan1999/CVPR2023_P-MIL)
