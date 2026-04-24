---
title: 流匹配与扩散模型
tags: 机器学习
sidebar:
  nav: cs
mathjax: true
key: 2026-04-24-1
aside:
    toc: true
---

很强劲啊 又开新坑

<!--more-->

### 生成式AI
从数据中生成样本的定义如下：
![20260424171016](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260424171016.png)

条件化输出：
将输入的条件定义为y，希望这个变量y条件化生成过程
引入条件数据分布 $p_{\mathrm{data}}(\cdot \mid y)$, 点为占位符，该表达式意味着：给定这个提示词y，给定这个提示词后的数据分布是什么
条件化生成即从这个条件分布中采样：
![20260424171810](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260424171810.png)
在生成模型中，通常我们有的是初始分布$p_{init}$,可以视作高斯分布，均值为0，以对角单位矩阵作为协方差矩阵，而生成模型从初始分布中取样（大多为白噪声），最后输出一个数据矩阵，可以看作为生成模型将初始分布转换为数据分布
![20260424172239](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260424172239.png)