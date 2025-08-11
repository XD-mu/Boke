---
title: LLM为什么Decoer-only架构呢
copyright: true
mathjax: true
date: 2025-08-11
categories:
- LLM基础
tags: 
---

Transformer模型一开始就是用来做seq2seq任务的，所以它包含Encoder和Deocer两个部分；这两者的区别主要是：**Encoder再抽取序列中的某一个词的特征时能够看到整个序列的信息，也就是说上下文同时看到，但Decoer中因为存在Mask机制，使得在编码某一个词的特征的时候只能看到自己和它前文的文本信息**

目前来说主要的几种架构是：
- 以BERT为代表的encoder-only
- 以T5和BART为代表的encoder-decoder
- 以GPT为代表的decoer-only
- 以UNILM9为代表的PrefixLM（相比GPT只改了attention mask，前缀部分是双向，后面要生成的部分是单向的causal mask）

然后说明要比较的对象: 首先淘汰掉BERT这种encoder-only，因为**它用masked language modeling预训练，不擅长做生成任务**，做NLUQ一般也需要有监督的下游数据微调: 相比之下decoder-only的模型用next token prediction预训练，兼顾理解和生成，在各种下游任务上的zero-shot和few-shot泛化性能都很好。为啥引入了一部分双向attention的encoder-decoder和Prefix-LM没有被大部分大模型工作采用? (它们也能兼顾理解和生成，泛化性能也不错)

# 1. Encoder 的低秩问题
LLM之所以主要都用Decoder-only架构，除了训练效率和工程实现上的优势外，在理论上是因为Encoder的双向注意力会存在低秩问题，这可能会削弱模型表达能力，就生成任务而言，引入双向注意力并无实质好处。而Encoder-Decoder架构之所以能够在某些场景下表现更好，大概只是因为它多了一倍参数。所以，在同等参数量、同等推理成本下，Decoder-only架构就是最优选择了。（详细看：[为什么现在的LLM都是Decoder-only的架构？](https://kexue.fm/archives/9529)）

# 2. 更好的Zero-Shot性能，让它更适合大语料自监督学习

直接说结论：**decoder-only 模型在没有任何 tuning 数据的情况下、zero-shot 表现最好，而 encoder-decoder 则需要在一定量的标注数据上做 multitask finetuning 才能激发最佳性能。**

目前的LLM的训练范式还是在大规模语料上做自监督学习，很显然，Zero-Shot性能更好的decoder-only架构才能更好地利用这些无标注数据。此外，Instruct GPT在自监督学习外还引入了RLHF作辅助学习。RLHF本身也不需要人工提供任务特定的标注数据，仅需要在LLM生成的结果上作排序。虽然目前没有太多有关RLHF + encoder-decoder的相关实验，直觉上RLHF带来的提升可能还是不如multitask finetuning，毕竟前者本质只是ranking、引入监督信号没有后者强。

# 3. 效率
decoder-only支持一直复用KV-Cache，对多轮对话更友好，因为每个Token的表示和它之前的输入有关系，而encoder-decoder和PrefixLM就很难做到。