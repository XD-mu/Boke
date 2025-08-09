---
title: Learning Action Completeness from Points for Weakly-supervised Temporal Action Localization
copyright: true
mathjax: true
date: 2023-12-13 15:07:17
categories:
- 课内学习
- 论文精读
tags:
- 视频定位;
- 弱监督;
- 时序动作定位;
- 单帧
---

## 1.Abstract

​	我们要解决的问题是，在每个动作实例只有一个帧标签的情况下，如何定位动作的时间间隔，以进行训练。由于标签稀疏，现有工作无法学习动作的完整性，从而导致零碎的动作预测。在本文中，我们提出了一个新颖的框架，即生成密集的伪标签，为模型提供完整性指导。具体来说，我们首先选择伪背景点来补充点级动作标签。然后，通过将点作为种子，我们搜索可能包含完整动作实例的最佳序列，同时与种子达成一致。为了从获得的序列中学习完整性，我们引入了两种新的损失，分别从动作得分和特征相似性方面对动作实例和背景实例进行对比。实验结果表明，我们的完整性指导确实有助于模型定位完整的动作实例，从而大幅提高了性能，尤其是在高 IoU 阈值下。此外，我们还在四个基准测试中证明了我们的方法优于现有的先进方法： THUMOS'14、GTEA、BEOID 和 ActivityNet。值得注意的是，我们的方法甚至可以与最新的全监督方法相媲美，而注释成本却低 6 倍。



<!-- more -->

## 2.Introduction

​	



## 3.Related Work





## 4.Model





## 5.Experiment

​	



## 6.Conclusion





## 7.相关链接

​	论文地址：

​	作者源码地址：

​	Pytorch源码版本：
