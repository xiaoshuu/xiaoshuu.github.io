---
title: 图机器学习cs224
tags: 机器学习
sidebar:
  nav: cs
mathjax: true
key: 2026-01-27-1
aside:
    toc: true
---

虽然之前的课都没学完

<!--more-->

#### 本课目标： 在深度学习中如何用图这种关系结构来做出更好、更准确的预测
并去掉传统机器学习里对原始数据的特征工程，直接从图学习什么是一个好的图表示

## Components of a Network
* Objects:nodes,vertices $N$
* Interactions:links,edges $E$
* System:network,graph $G(N,E)$

图是一种通用的语言，深度学习模型可以对图进行预测，无论节点对应的是什么事物

### 图的分类
![20260203214639](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260203214639.png)

#### 无向图

![20260304134309](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260304134309.png)

Node Degree（节点度）：给定节点的邻近边数量 <br/>
Avg.degree（平均节点度）：网络中所有节点度的平均值，数值为此网络中的边数除以节点数的两倍

#### 有向图

![20260304134626](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260304134626.png)

有向图中对in-degree（入度）和out-degree（出度）进行区分，入度指指向节点的边数，出度是节点向外指的边的数量

#### 二部图

![20260304135005](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260304135005.png)

二部图通常是包含两种不同类型节点的图，节点仅与其他类型的节点相互联系，图可被分为左右两个区域，且边只从左边的分区往右边连

##### 对二部图进行折叠/投影

![20260304135217](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260304135217.png)

在中间的二部图中（假设u为作者，v为其发表文章），作者123共同发表了一篇文章，故折叠后将其相连，34作者没有共同发表文章，故不相连。

### 图的表示

#### 矩阵表示
![20260304135536](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260304135536.png)
![20260304135624](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260304135624.png)
Most real-world networks are sparse, E<<$E_{max}$ <br/>
Consequence:Adjacency matrix is filled with zeros

#### 边列表

![20260304135939](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260304135939.png)

但很难对其做操作或进行分析

#### 邻接列表

![20260304140102](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260304140102.png)

Tips: 边和节点都可以带上属性，例如边可以有权重，节点可以有该物品的属性值
![20260304140346](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260304140346.png)
在图中也可以加上自循环或多边图（一对节点之间有多个边）
![20260304140604](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260304140604.png)

### 连通性

#### 无向图
![20260304140707](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260304140707.png)
![20260304140813](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260304140813.png)

#### 有向图
![20260304140906](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/  20260304140906.png)
![20260304140930](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260304140930.png)

### 如何对图进行预测

目标：对一组物品进行预测

设计选择：
* 特征：一个D维向量
* 对象：节点，边，节点集和整个图
* 目标函数：我们要预测的标签是什么

#### 单个节点的节点级任务和特性

目标：描述给定节点周围的网络结构，可描述的方法如下：
* Node degree(节点度)
* Node centrality(节点中心性)
本地网络结构表征，描述给定节点有多少条边，也包括它周围节点的结构
* Clustering coefficient(聚类系数)
* Graphlets(图元)

##### 节点度
一般指我们通过节点v的边数得知了它在网络中的结构，缺点是该方法不做区分地对待每个节点，无法对相同节点度的节点进行区分

##### 特征向量中心性
如果节点v被重要的许多相邻节点u包围的话，那它是十分关键的节点，故给定节点v的重要性，可以普遍用$1/{\lambda}$加上其在网络中邻近节点的重要度的总和表示，即朋友越重要，自己的重要性就越高
![20260414205403](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260414205403.png)
直观来看，节点的重要度是与该节点相连节点的重要度的归一化总和

##### 中介中心度
中介中心度指如果这个节点与其他节点对之间的路径大多数都是最短的，那么它就是重要节点
![20260414205914](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260414205914.png)

##### 接近中心度
如果一个节点到达网络中的其他节点的路径和是最短的，那么它是重要的
![20260414210148](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260414210148.png)

##### 聚类系数
聚类系数衡量一个节点与邻近节点之间的联系程度，节点 $v$ 的聚类系数，是指 $v$ 的邻居节点之间实际存在的边数，除以这些邻居之间可能形成的最大边数
![20260414213728](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260414213728.png)
分母：$C_k^2$
* 这里 $k$ 是节点 $v$ 的度（即 $v$ 有多少个邻居）
* 从 $k$ 个邻居中选出 2 个节点组成一对，总共有 $\frac{k(k-1)}{2}$ 种组合。这就是邻居之间理论上最多能产生的边数
分子：邻居节点之间的边数
* 看一看和$v$相连的节点有没有边相连
结果：
* 如果邻居之间全都是两两相连，系数就是1
* 如果邻居之间互不认识，系数就是0
![20260414214912](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260414214912.png)
观察发现，邻居间的边数基本上是节点周围网络中三角形的个数
$ 聚类系数 = \frac{实际存在的三角形数量}{理论上可能的最大三角形数量} $

##### 图元
图元是一个有根连接的非同构子图
![20260414220121](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260414220121.png)

##### 图元度向量（GDV）
GDV是基于图元的节点特征，指一个给定的图元以该给定节点为根出现的次数
![20260414220443](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260414220443.png)
