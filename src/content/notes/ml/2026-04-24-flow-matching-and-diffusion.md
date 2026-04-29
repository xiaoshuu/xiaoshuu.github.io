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
> 生成模型可以被理解为：从一个简单分布出发，通过某种动力系统，把样本变换到真实数据分布中。

换句话说，生成模型的核心问题是：

$$
p_{\mathrm{init}} \longrightarrow p_{\mathrm{data}}
$$

其中：

- $p_{\mathrm{init}}$ 是一个简单的初始分布，通常是高斯分布；
- $p_{\mathrm{data}}$ 是真实数据分布；
- 生成模型的任务是把从 $p_{\mathrm{init}}$ 中采样得到的噪声，逐渐变成像真实数据一样的样本。

从数据中生成样本的定义如下：
![20260424171016](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260424171016.png)

#### 1. 从“生成”到“采样”

生成式 AI 的目标是生成新的对象，例如图像、视频、蛋白质结构、三维形状等。

在数学上，我们首先把这些对象表示成向量：

$$
z \in \mathbb{R}^d
$$

例如：

- 一张图像可以看成一个由像素组成的高维向量；
- 一个视频可以看成多个图像帧组成的高维向量；
- 一个分子结构可以看成原子坐标组成的向量。

因此，所谓“生成一个对象”，在数学上可以理解为：

> 在高维空间中找到一个合理的点 $z$。

但什么叫“合理”？

如果我们想生成一张狗的图片，那么有些图片明显不像狗，有些图片虽然像动物但不是狗，有些图片则非常自然、合理。  
为了把这个直觉形式化，我们引入数据分布。

#### 2. 数据分布

真实世界中的数据并不是均匀分布在整个高维空间里的。  
例如，随机生成一个像素矩阵，大概率只会得到噪声，而不是一张自然图像。

我们用数据分布表示真实数据在空间中的分布情况：

$$
p_{\mathrm{data}}(z)
$$

它表示：

> 在真实数据中，样本 $z$ 出现的可能性有多大。

如果一个样本看起来很自然、很符合真实数据，那么它在 $p_{\mathrm{data}}$ 下的概率密度就比较高。  
如果一个样本只是随机噪声，那么它在 $p_{\mathrm{data}}$ 下的概率密度就很低。

因此，生成模型的目标可以写成：

$$
z \sim p_{\mathrm{data}}
$$

也就是说：

> 生成一个对象，本质上就是从真实数据分布中采样。

不过需要注意的是，在实际问题中，我们并不知道完整的 $p_{\mathrm{data}}$。  
我们拥有的只是一个有限的数据集：

$$
\{z_1, z_2, \dots, z_N\}
$$

这个数据集可以看作是从真实数据分布中采样得到的一批样本。

#### 3. 条件生成

很多生成任务不是随便生成一个样本，而是根据某个条件生成样本。

例如：

- 给定提示词 “dog”，生成狗的图片；
- 给定提示词 “cat”，生成猫的图片；
- 给定提示词 “landscape”，生成风景图。

我们把条件记为$y$

此时，模型不再是从整体数据分布 $p_{\mathrm{data}}(z)$ 中采样，而是从条件数据分布中采样：

$$
p_{\mathrm{data}}(z \mid y)
$$

它表示：

> 在给定条件 $y$ 的情况下，样本 $z$ 的分布是什么。

有时也会写成：

$$
p_{\mathrm{data}}(\cdot \mid y)
$$

这里的 $\cdot$ 是占位符，表示这个分布作用在所有可能的 $z$ 上。

条件生成的目标就是：

$$
z \sim p_{\mathrm{data}}(z \mid y)
$$

例如，当 $y=\text{“dog”}$ 时，我们希望采样得到的是狗的图片；  
当 $y=\text{“cat”}$ 时，我们希望采样得到的是猫的图片。

在生成模型中，通常我们有的是初始分布$p_{init}$,可以视作高斯分布，均值为0，以对角单位矩阵作为协方差矩阵，而生成模型从初始分布中取样（大多为白噪声），最后输出一个数据矩阵，可以看作为生成模型将初始分布转换为数据分布
![20260424172239](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260424172239.png)

#### 4. 生成模型的基本形式

虽然我们的目标是从 $p_{\mathrm{data}}$ 中采样，但真实的数据分布通常非常复杂，我们无法直接采样。

因此，生成模型通常采用一个间接的方法：

1. 先从一个简单分布中采样；
2. 再通过模型把这个简单样本变换成真实数据样本。

这个简单分布记作：

$$
p_{\mathrm{init}}
$$

通常取为标准高斯分布：

$$
p_{\mathrm{init}} = \mathcal{N}(0, I)
$$

也就是说，我们先采样一个噪声：

$$
x_0 \sim p_{\mathrm{init}}
$$

然后通过生成模型得到：

$$
x_1 \sim p_{\mathrm{data}}
$$

因此，生成模型可以理解为一个分布变换过程：

$$
p_{\mathrm{init}} \longrightarrow p_{\mathrm{data}}
$$

这也是 Flow Models 和 Diffusion Models 的共同出发点。


#### 5. Flow Models

###### 5.1 Flow Model 的核心思想

Flow Model 的基本思想是：

> 用一个连续时间的运动过程，把初始噪声逐渐移动到数据分布中。

我们用 $X_t$ 表示时间 $t$ 时刻的样本状态，其中：

$$
t \in [0,1]
$$

当 $t=0$ 时：

$$
X_0 \sim p_{\mathrm{init}}
$$

当 $t=1$ 时：

$$
X_1 \sim p_{\mathrm{data}}
$$

所以整个生成过程可以写成：

$$
X_0 \rightarrow X_t \rightarrow X_1
$$

直观理解就是：

> 一个点从噪声分布出发，沿着某条路径运动，最后到达数据分布。

##### 5.2 轨迹

轨迹描述的是一个样本随时间变化的路径。

如果给定初始点 $X_0=x_0$，那么随着时间变化，这个点会形成一条曲线：

$$
X = \{X_t\}_{t \in [0,1]}
$$

也可以理解为一个函数：

$$
t \mapsto X_t
$$

其中：

- $X_0$ 是初始状态；
- $X_t$ 是中间状态；
- $X_1$ 是最终状态。

所以，轨迹回答的问题是：

> 一个具体样本从起点到终点是怎么走的？

##### 5.3 向量场

为了描述样本如何运动，我们需要知道每个位置、每个时刻的运动方向和速度。

这就需要引入向量场。

向量场通常写作：

$$
u_t(x)
$$

它表示：

> 在时间 $t$，如果样本位于位置 $x$，那么它应该以什么速度、朝什么方向运动。

也就是说，向量场是一个函数：

$$
(x,t) \mapsto u_t(x)
$$

输入：

- 当前的位置 $x$；
- 当前的时间 $t$。

输出：

- 一个和 $x$ 维度相同的向量；
- 表示当前位置的速度方向和大小。

这里需要注意：

> 向量场不是一条轨迹，而是定义在整个空间上的运动规则。

轨迹是某一个点按照这个规则走出来的路径；  
向量场则规定了所有点在所有时刻应该怎么运动。

##### 5.4 ODE 表示

Flow Model 中的运动过程可以用常微分方程 ODE 表示：

$$
\frac{dX_t}{dt} = u_t(X_t)
$$

其中：

- $X_t$ 是当前样本的位置；
- $u_t(X_t)$ 是当前位置和当前时间下的速度；
- $\frac{dX_t}{dt}$ 表示样本随时间变化的速度。

如果给定初始条件：（即沿着向量场的指定方向前进）

$$
X_0 = x_0
$$

那么求解这个 ODE，就可以得到整条轨迹：

$$
X_0, X_t, X_1
$$

在生成模型中，我们关心的是最终结果 $X_1$。


##### 5.5 Flow Map
对于每一个初始点 $x_0$，ODE 都会给出一个对应的运动结果。，Flow map 可以看作由向量场产生的一族映射：

$$
\psi:\mathbb{R}^d \times [0,1] \to \mathbb{R}^d
$$

$$
(x_0,t) \mapsto \psi_t(x_0)
$$

其中，$x_0$ 是初始点，$t$ 是时间，$\psi_t(x_0)$ 表示从初始点 $x_0$ 出发，沿着向量场运动到时间 $t$ 时的位置。它表示：

> 如果一个点从 $x_0$ 出发，沿着向量场运动到时间 $t$，它会到达哪里。

当 $t=0$ 时，点还没有移动，所以：

$$
\psi_0(x_0)=x_0
$$

Flow map 由向量场 $u_t$ 决定，它满足下面的微分方程：

$$
\frac{d}{dt}\psi_t(x_0)=u_t(\psi_t(x_0))
$$

这个式子的含义是：  
从 $x_0$ 出发的点在时间 $t$ 的运动速度，等于向量场 $u_t$ 在当前位置 $\psi_t(x_0)$ 处给出的速度。

因此，如果记：

$$
X_t=\psi_t(x_0)
$$

那么上式就可以写成更常见的形式：

$$
\frac{dX_t}{dt}=u_t(X_t)
$$

流可以被看做是针对许多初始条件的常微分方程解的集合，总体关系为向量场定义了常微分方程，而轨迹是这个常微分方程的一个解，流是不同初始条件下的轨迹集合

##### 5.5.5 ODE的存在唯一性

为了让 Flow Model 是良好定义的，我们需要保证：  
从任意初始点 $x_0$ 出发，ODE 都存在唯一解。

Picard–Lindelöf 定理说明，如果向量场 $u_t(x)$ 足够光滑，例如连续可微且导数有界，那么这个 ODE 存在唯一解。

更一般地，如果向量场满足 Lipschitz 条件：

$$
\|u_t(x)-u_t(y)\| \leq L\|x-y\|
$$

那么 ODE 也有唯一解。

这意味着，从同一个初始点 $x_0$ 出发，轨迹是唯一确定的。因此 flow map

$$
\psi_t(x_0)
$$

是良好定义的。

也就是说：

$$
\psi_t(x_0)=X_t
$$

表示从初始点 $x_0$ 出发，沿着向量场运动到时间 $t$ 时的位置。

> 只要向量场足够规整，从每个初始点出发的 ODE 轨迹就存在且唯一，因此 Flow Model 中的 flow map 是可以被正确定义的。

##### 5.6 Flow Model 如何生成样本

Flow Model 的采样过程如下：

1. 从初始分布中采样一个噪声点：

$$
X_0 \sim p_{\mathrm{init}}
$$

2. 按照 ODE 进行演化：

$$
\frac{dX_t}{dt} = u_t(X_t)
$$

3. 从 $t=0$ 积分到 $t=1$；

4. 返回最终样本：

$$
X_1
$$

这就是 Flow Model 的生成过程。

##### 5.7 数值求解：Euler Method

在实际计算机中，我们无法真正连续地求解 ODE，因此需要离散化。

最简单的方法是 Euler Method。

设步长为：

$$
h = \frac{1}{n}
$$

如果当前时刻是 $t$，当前位置是 $X_t$，那么下一步可以近似为：

$$
X_{t+h} = X_t + h u_t(X_t)
$$

直观理解：

> 在当前位置，看一下向量场指向哪里，然后沿这个方向前进一小步。

重复很多次之后，就可以从 $X_0$ 走到 $X_1$。

#### 6. Diffusion Models

##### 6.1 从 ODE 到 SDE

Flow Model 使用的是确定性的 ODE：

$$
dX_t = u_t(X_t)dt
$$

如果初始点确定，那么轨迹也是确定的。

Diffusion Model 则在这个过程中加入随机性，也就是把 ODE 扩展为 SDE（随机微分方程）：

$$
dX_t = u_t(X_t)dt + \sigma_t dW_t
$$

其中：

- $u_t(X_t)dt$ 是确定性的运动部分，即向量场；
- $dW_t$ 是随机噪声，详细见6.2；
- $\sigma_t$ 是 diffusion coefficient，也就是扩散系数， $\sigma_t$控制随机噪声的强度。
$\boxed{
\text{drift coefficient } u_t \text{ 控制方向，diffusion coefficient } \sigma_t \text{ 控制随机性}
}$

当：

$$
\sigma_t = 0
$$

时，SDE 就退化成 ODE：

$$
dX_t = u_t(X_t)dt
$$

因此可以理解为：

> Flow Model 是没有随机噪声的情形，Diffusion Model 是带随机噪声的情形。

任何在时间上连续且具有连续轨迹的马尔可夫过程都可以使用该形式的随机微分方程表示

##### 6.2 Brownian Motion

SDE 中的随机性通常来自 Brownian Motion，也叫 Wiener Process。

Brownian Motion 记作：

$$
W_t
$$

在时间为0时为0，它有两个重要性质。

第一，增量服从高斯分布：

$$
W_{t+h} - W_t \sim \mathcal{N}(0, hI)
$$

也就是说，时间间隔越大，随机变化的方差越大，方差基本沿时间线性增长。

第二，不同时间段的增量相互独立。

直观来说，Brownian Motion 描述的是一种连续时间中的随机游走。

##### 6.3 SDE 的采样：Euler-Maruyama Method

和 ODE 类似，SDE 也需要用数值方法求解。

常用方法是 Euler-Maruyama Method。

对于 SDE：

$$
dX_t = u_t(X_t)dt + \sigma_t dW_t
$$

离散化后可以写成：

$$
X_{t+h} = X_t + h u_t(X_t) + \sigma_t \sqrt{h}\epsilon
$$

其中：

$$
\epsilon \sim \mathcal{N}(0, I)
$$

这里可以看到，SDE 的更新比 ODE 多了一项随机噪声：

$$
\sigma_t \sqrt{h}\epsilon
$$

如果 $\sigma_t=0$，这一项消失，就回到了 Euler Method：

$$
X_{t+h} = X_t + h u_t(X_t)
$$

##### 6.4 Diffusion Model 如何生成样本

Diffusion Model 的生成过程可以理解为：

1. 从初始分布中采样：

$$
X_0 \sim p_{\mathrm{init}}
$$

2. 按照 SDE 演化：

$$
dX_t = u_t(X_t)dt + \sigma_t dW_t
$$

3. 使用 Euler-Maruyama Method 逐步模拟；

4. 最终得到：

$$
X_1 \sim p_{\mathrm{data}}
$$

在机器学习中，通常会用神经网络来参数化其中的向量场：

$$
u_\theta(x,t)
$$

也就是说，模型要学习的是：

> 在每个时间、每个位置，样本应该如何运动。

扩散系数 $\sigma_t$ 通常是预先设定的，而神经网络主要负责学习运动方向。

#### 7. Flow Model 与 Diffusion Model 的对比

| 模型 | 数学形式 | 是否随机 | 数值方法 | 核心思想 |
|---|---|---|---|---|
| Flow Model | ODE | 否 | Euler Method / ODE Solver | 沿确定性向量场从噪声走到数据 |
| Diffusion Model | SDE | 是 | Euler-Maruyama Method | 在确定性运动基础上加入随机扩散 |
| 关系 | 当 $\sigma_t=0$ 时，SDE 退化为 ODE | - | - | Flow Model 可看作 Diffusion Model 的特殊情况 |

#### 8. 本讲的核心理解

第一讲最重要的主线是：

> 生成模型不是凭空“画”出一个样本，而是把一个简单分布中的样本，通过某种动力系统，变换成数据分布中的样本。

这个动力系统可以有两种形式。

第一种是 ODE，对应 Flow Model：

$$
\frac{dX_t}{dt} = u_t(X_t)
$$

第二种是 SDE，对应 Diffusion Model：

$$
dX_t = u_t(X_t)dt + \sigma_t dW_t
$$

二者的共同目标都是：

$$
p_{\mathrm{init}} \longrightarrow p_{\mathrm{data}}
$$

也就是从简单分布生成复杂数据分布。

#### 9. 容易混淆的地方

##### 9.1 轨迹和向量场不是一个东西

轨迹是一个具体样本走出来的路径：

$$
t \mapsto X_t
$$

向量场是整个空间中的运动规则：

$$
(x,t) \mapsto u_t(x)
$$

一个向量场可以产生无数条轨迹。  
每一个不同的初始点，都可以沿着同一个向量场生成一条不同的轨迹。

##### 9.2 Flow 不是单独一条曲线

Flow 更准确地说是由 ODE 产生的一族映射：

$$
\phi_t(x_0)
$$

它告诉我们：

> 从任意初始点 $x_0$ 出发，经过时间 $t$ 后会到哪里。

所以，Flow 描述的是整个空间如何随时间流动，而不是某一个点的单独运动。

##### 9.3 $p_{\mathrm{data}}$ 通常是未知的

我们想从 $p_{\mathrm{data}}$ 中采样，但实际上并不知道这个分布的完整形式。

我们只有数据集：

$$
\{z_1,z_2,\dots,z_N\}
$$

模型训练的目的就是利用这些有限样本，学习一个能够近似生成真实数据分布的过程。

### 生成式AI的训练目标

#### 区分Conditional 和 Marginal
- > conditional表示围绕某一个具体数据点$z$的情况
- marginal表示对整个数据分布平均之后的整体情况

#### 1. Probability Path：从噪声分布到数据分布的路径
生成模型的目标是把简单分布变成数据分布：
$$
p_{\mathrm{init}} \longrightarrow p_{\mathrm{data}}
$$
但要训练模型，我们不能只说起点和终点，还需要知道中间过程应该是什么样的。

因此，引入一条随时间变化的概率分布路径：$p_t$
其中t∈[0,1],并且希望满足：
$$
p_0 = p_{\mathrm{init}}
$$ 

$$
p_1 = p_{\mathrm{data}}
$$
这条路径称为 probability path。

它描述的是：

> 在每个时间 $t$，样本整体应该服从什么分布。

直观地说，$p_t$ 是从噪声分布到数据分布的一条连续变化路径。
$p_t$是一种分布，可以从中取样

#### 2.Conditional Probability Path
##### 2.1 定义
给定一个真实数据点：$z \sim p_{\mathrm{data}}$
我们可以构造一条围绕这个数据点 $z$ 的条件概率路径：$p_t(\cdot \mid z)$
它表示：
> 在给定目标数据点 $z$ 的情况下，时间 $t$ 时中间样本的分布。

这里的 $\cdot$ 是占位符，表示这个分布作用在所有可能的 $x$ 上。

也可以写成：$p_t(x \mid z)$
表示时间 $t$ 时，样本位于 $x$ 附近的概率密度。

它的作用是：

> 把初始分布 $p_{\mathrm{init}}$ 插值到某一个具体数据点 $z$。

##### 2.2 高斯条件路径
式子如下： 

$$
p_t(\cdot \mid z)=\mathcal{N}(\alpha_t z,\beta_t^2 I_d)
$$
这表示在时间 $t$，样本服从一个高斯分布，其中：
- 均值是 $\alpha_t z$；
- 方差是 $\beta_t^2 I_d$；
- $I_d$ 是 $d$ 维单位矩阵；
- $\alpha_t$ 控制数据点 $z$ 的比例；
- $\beta_t$ 控制噪声强度。

等价地，可以写成采样形式：
$$
X_t = \alpha_t z + \beta_t \epsilon,\epsilon \sim \mathcal{N}(0,I_d)
$$

为了让它从噪声走到数据点，我们通常希望：$\alpha_0 = 0,\qquad \beta_0 = 1$,这样$X_0 = \epsilon \sim \mathcal{N}(0,I_d)$也就是初始噪声。

同时希望：$\alpha_1 = 1,\qquad \beta_1 = 0$,这样有$X_1 = z$,也就是最终到达数据点本身。

所以 Gaussian conditional probability path 可以理解为：
> 用一个逐渐移动、逐渐收缩的高斯分布，把噪声分布变成某一个具体数据点。

#### 3.Marginal(边缘) Probability Path

前面我们先构造了 conditional probability path：

$$
p_t(x \mid z)
$$

它表示：给定某一个数据点 $z$ 时，时间 $t$ 的中间样本分布。

但是生成模型最终不是只针对某一个固定的数据点 $z$，而是要生成整个数据分布：

$$
p_{\mathrm{data}}
$$

所以我们需要把所有数据点对应的 conditional path 混合起来，得到 marginal probability path：

$$
p_t(x)=\int p_t(x\mid z)p_{\mathrm{data}}(z)\,dz
$$
即对所有可能的数据点 $z$，把它们对应的路径 $p_t(x\mid z)$ 按照 $p_{\mathrm{data}}(z)$ 加权平均。

这里的构造过程可以理解为：

1. 先从真实数据分布中采样一个数据点：

$$
z \sim p_{\mathrm{data}}
$$

2. 在给定这个数据点 $z$ 的条件下，从 conditional path 中采样：

$$
X_t \mid z \sim p_t(\cdot \mid z)
$$

3. 如果不再关心具体采到了哪个 $z$，只看 $X_t$ 本身的分布，那么：

$$
X_t \sim p_t
$$

这个 $p_t$ 就是 marginal probability path。

也就是说：

$$
z \sim p_{\mathrm{data}}, \qquad X_t \mid z \sim p_t(\cdot \mid z)
\quad \Longrightarrow \quad
X_t \sim p_t
$$

其中：

$$
p_t(x)=\int p_t(x\mid z)p_{\mathrm{data}}(z)\,dz
$$

我们希望这条 marginal probability path 满足：

$$
p_0 = p_{\mathrm{init}}
$$

$$
p_1 = p_{\mathrm{data}}
$$

也就是说，在 $t=0$ 时，它是初始噪声分布；在 $t=1$ 时，它变成真实数据分布。

对于 Gaussian conditional path：

$$
p_t(x\mid z)=\mathcal{N}(\alpha_t z,\beta_t^2 I_d)
$$

如果满足：

$$
\alpha_0=0,\qquad \beta_0=1
$$

那么：

$$
p_0(x\mid z)=\mathcal{N}(0,I_d)
$$

它与 $z$ 无关，因此混合后仍然是：

$$
p_0=\mathcal{N}(0,I_d)=p_{\mathrm{init}}
$$

如果满足：

$$
\alpha_1=1,\qquad \beta_1=0
$$

那么：

$$
p_1(x\mid z)
$$

会集中到数据点 $z$ 上。

把所有数据点 $z\sim p_{\mathrm{data}}$ 混合起来后，就得到：

$$
p_1=p_{\mathrm{data}}
$$

因此，marginal probability path 的作用是：

> 把针对单个数据点的 conditional path，变成针对整个数据分布的路径。

#### 4.Conditional Vector Field

##### 4.1 Conditional Vector Field 的定义

前面我们已经构造了 conditional probability path：

$$
p_t(x \mid z)
$$

它表示：在给定某个数据点 $z$ 的情况下，时间 $t$ 时中间样本 $x$ 的分布。

现在我们希望找到一个向量场：

$$
u_t^{\mathrm{target}}(x \mid z)
$$

这个向量场叫作 **conditional vector field**。

它是一个依赖于数据点 $z$ 的速度场，可以理解为一个函数：

$$
(x,t,z) \mapsto u_t^{\mathrm{target}}(x \mid z)
$$

其中：

- $x$ 是当前样本的位置；
- $t$ 是当前时间；
- $z$ 是给定的目标数据点；
- $u_t^{\mathrm{target}}(x \mid z)$ 是当前位置 $x$ 在时间 $t$ 下应该具有的速度。

所以 conditional vector field 的含义是：

> 给定目标数据点 $z$，如果当前样本位于 $x$，时间为 $t$，那么它应该以什么速度运动。

它和普通 vector field 的区别在于，普通 vector field 写作：

$$
u_t(x)
$$

只依赖当前位置 $x$ 和时间 $t$。

而 conditional vector field 写作：

$$
u_t(x \mid z)
$$

它还依赖于目标数据点 $z$。

也就是说，不同的 $z$ 会对应不同的速度场。

##### 4.2 Conditional Vector Field 应该满足什么性质？

conditional vector field 需要满足下面这个性质：

如果初始样本来自 conditional path 的起点：

$$
X_0 \sim p_0(\cdot \mid z)
$$

并且样本按照 ODE 运动：

$$
\frac{d}{dt}X_t=u_t^{\mathrm{target}}(X_t \mid z)
$$

那么在任意时间 $t$，样本分布都应该满足：

$$
X_t \sim p_t(\cdot \mid z)
$$

也就是说，这个向量场需要让 ODE 的分布正好沿着 conditional probability path 走。

因此可以把 conditional vector field 定义为：

> 能够通过 ODE 把条件路径 $p_t(\cdot \mid z)$ 实现出来的速度场。

换句话说，conditional probability path 告诉我们：

$$
\text{每个时间 } t \text{ 的分布应该是什么}
$$

而 conditional vector field 告诉我们：

$$
\text{样本应该怎么运动，才能产生这条分布路径}
$$

##### 4.3 从分布角度看 Conditional Vector Field

更严格地说，如果一个向量场 $u_t^{\mathrm{target}}(x \mid z)$ 能够产生 conditional probability path $p_t(x\mid z)$，那么它们应该满足 conditional continuity equation：

$$
\frac{\partial}{\partial t}p_t(x \mid z)=-\operatorname{div}\left(p_t(x \mid z)u_t^{\mathrm{target}}(x \mid z)\right)
$$

这个方程的意思是：

> 条件分布 $p_t(x\mid z)$ 的变化，是由向量场 $u_t^{\mathrm{target}}(x\mid z)$ 搬运概率质量造成的。

所以 conditional vector field 的作用可以总结为：

$$
u_t^{\mathrm{target}}(x \mid z)
\quad
\Longrightarrow
\quad
p_t(x \mid z) \text{ 按照这条路径演化}
$$

##### 4.4 直观理解

可以把每个数据点 $z$ 想象成一个目标点。

对于每个目标点 $z$，我们都构造一条 conditional probability path：

$$
p_t(\cdot \mid z)
$$

这条路径描述了：

> 一团噪声如何逐渐移动并收缩到数据点 $z$ 附近。

而 conditional vector field：

$$
u_t^{\mathrm{target}}(x \mid z)
$$

描述的是：

> 在这条路径上，每个中间点 $x$ 应该往哪里走。

所以：

- conditional probability path 描述“分布应该长什么样”；
- conditional vector field 描述“样本应该怎么动”。

一句话理解：

$$
\boxed{
\text{Conditional vector field 是实现 conditional probability path 的速度场。}
}
$$

##### 4.5 Gaussian Path 下的 Conditional Vector Field

对于Gaussian path有
$$
X_t = \alpha_t z + \beta_t \epsilon
$$
其中：$\epsilon \sim \mathcal{N}(0,I_d)$
对时间 $t$ 求导：
$$
\frac{d}{dt}X_t=\dot{\alpha}_t z + \dot{\beta}_t \epsilon
$$
但是向量场应该写成关于当前状态 $x$ 和数据点 $z$ 的函数，而不是关于 $\epsilon$ 的函数。
由$x = \alpha_t z + \beta_t \epsilon$可得$\epsilon = \frac{x-\alpha_t z}{\beta_t}$
代回去:$\frac{d}{dt}X_t=\dot{\alpha}_t z+\dot{\beta}_t\frac{x-\alpha_t z}{\beta_t}$
展开：$\frac{d}{dt}X_t=\dot{\alpha}_t z+\frac{\dot{\beta}_t}{\beta_t}x-\frac{\dot{\beta}_t}{\beta_t}\alpha_t z$
整理得到:$u_t^{\mathrm{target}}(x \mid z)=\left(\dot{\alpha}_t-\frac{\dot{\beta}_t}{\beta_t}\alpha_t
\right)z
+
\frac{\dot{\beta}_t}{\beta_t}x$
该式子可以理解为：

当前点 $x$ 的速度由两部分决定：一部分和目标数据点 $z$ 有关，另一部分和当前位置 $x$ 有关。

其中：$\left(\dot{\alpha}_t-\frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z$
表示把样本往数据点 $z$ 的方向推。

而：$\frac{\dot{\beta}_t}{\beta_t}x$
表示根据噪声尺度 $\beta_t$ 的变化，对当前位置做收缩或扩张。

如果 $\beta_t$ 逐渐变小，那么这个过程会把围绕数据点的高斯分布逐渐压缩，最后集中到 $z$ 附近。

#### 5. Marginal Vector Field

##### 5.1 为什么需要 Marginal Vector Field？

前面我们定义了 conditional vector field：

$$
u_t^{\mathrm{target}}(x \mid z)
$$

它表示：

> 给定某一个数据点 $z$ 时，当前点 $x$ 在时间 $t$ 应该以什么速度运动。

但是在真正生成的时候，我们并没有一个固定的目标数据点 $z$。

生成时只有：

$$
X_0 \sim p_{\mathrm{init}}
$$

然后希望通过一个 ODE：

$$
\frac{d}{dt}X_t = u_t^{\mathrm{target}}(X_t)
$$

最终得到：

$$
X_1 \sim p_{\mathrm{data}}
$$

所以 Flow Model 真正需要的不是依赖某个数据点 $z$ 的 conditional vector field，而是一个整体的速度场：

$$
u_t^{\mathrm{target}}(x)
$$

这个整体速度场就叫作 **marginal vector field**。

##### 5.2 Marginal Vector Field 的定义

Marginal vector field 定义为：

$$
u_t^{\mathrm{target}}(x)=
\int
u_t^{\mathrm{target}}(x \mid z)
\frac{
p_t(x \mid z)p_{\mathrm{data}}(z)
}{
p_t(x)
}
dz
$$

其中：

$$
p_t(x)=
\int p_t(x \mid z)p_{\mathrm{data}}(z)dz
$$

是 marginal probability path。

这个公式的含义是：

> marginal vector field 是所有 conditional vector field 的加权平均。

也就是说，在当前点 $x$ 处，可能有很多个数据点 $z$ 的 conditional path 会经过这里。  
每个数据点 $z$ 都会给出一个速度：

$$
u_t^{\mathrm{target}}(x \mid z)
$$

marginal vector field 就是把这些可能速度按照权重平均起来。

##### 5.3 权重是什么意思？

公式中的权重是：

$$
\frac{
p_t(x \mid z)p_{\mathrm{data}}(z)
}{
p_t(x)
}
$$

根据贝叶斯公式，它可以理解为：

$$
p_t(z \mid x)
$$

也就是：

> 在时间 $t$，观察到当前中间点 $x$ 时，它可能来自哪个数据点 $z$。

因此，marginal vector field 可以写成条件期望的形式：

$$
u_t^{\mathrm{target}}(x)=
\mathbb{E}
\left[
u_t^{\mathrm{target}}(x \mid z)
\mid x
\right]
$$

这句话很重要。

它表示：

> marginal vector field 是 conditional vector field 在给定当前点 $x$ 后，对所有可能数据来源 $z$ 的平均。

##### 5.4 直观理解

可以这样想：

每个数据点 $z$ 都有一条自己的 conditional probability path：

$$
p_t(x \mid z)
$$

也有对应的 conditional vector field：

$$
u_t^{\mathrm{target}}(x \mid z)
$$

如果当前中间点是 $x$，它可能来自很多不同的数据点：

$$
z_1,z_2,z_3,\dots
$$

对于每个可能的 $z_i$，都有一个“建议速度”：

$$
u_t^{\mathrm{target}}(x \mid z_i)
$$

但是生成模型不知道当前点 $x$ 具体应该走向哪个训练样本，所以它需要综合这些建议。

因此：

$$
u_t^{\mathrm{target}}(x)
$$

就是所有可能 conditional velocity 的平均结果。

简单说：

```text
conditional vector field：给定某个 z 时应该怎么走

marginal vector field：不知道具体 z 时，综合所有可能 z 后应该怎么走
```

##### 5.5 Marginal Vector Field 的推导

已知每个 conditional path 满足 conditional continuity equation：

$$
\frac{\partial}{\partial t}p_t(x \mid z)=
-\operatorname{div}
\left(
p_t(x \mid z)
u_t^{\mathrm{target}}(x \mid z)
\right)
$$

marginal path 定义为：

$$
p_t(x)=
\int p_t(x \mid z)p_{\mathrm{data}}(z)dz
$$

对时间求导：

$$
\frac{\partial}{\partial t}p_t(x)=
\int
\frac{\partial}{\partial t}p_t(x \mid z)
p_{\mathrm{data}}(z)dz
$$

代入 conditional continuity equation：

$$
\frac{\partial}{\partial t}p_t(x)=
-\int
\operatorname{div}
\left(
p_t(x \mid z)
u_t^{\mathrm{target}}(x \mid z)
\right)
p_{\mathrm{data}}(z)dz
$$

把散度放到积分外：

$$
\frac{\partial}{\partial t}p_t(x)=
-\operatorname{div}
\left(
\int
p_t(x \mid z)
u_t^{\mathrm{target}}(x \mid z)
p_{\mathrm{data}}(z)dz
\right)
$$

我们希望 marginal path 也满足 continuity equation：

$$
\frac{\partial}{\partial t}p_t(x)=
-\operatorname{div}
\left(
p_t(x)u_t^{\mathrm{target}}(x)
\right)
$$

所以需要：

$$
p_t(x)u_t^{\mathrm{target}}(x)=
\int
p_t(x \mid z)
u_t^{\mathrm{target}}(x \mid z)
p_{\mathrm{data}}(z)dz
$$

两边除以 $p_t(x)$：

$$
u_t^{\mathrm{target}}(x)=
\int
u_t^{\mathrm{target}}(x \mid z)
\frac{
p_t(x \mid z)p_{\mathrm{data}}(z)
}{
p_t(x)
}
dz
$$

这就是 marginal vector field 的公式。

#### 6. Continuity Equation
##### 6.1 ODE 如何影响概率分布？

Flow Model 中，单个样本按照 ODE 运动：

$$
\frac{d}{dt}X_t = u_t(X_t)
$$

但是训练生成模型时，我们关心的不是单个点，而是整个分布：

$$
X_t \sim p_t
$$

因此需要一个方程描述：

> 当所有点都按照向量场 $u_t$ 运动时，概率密度 $p_t(x)$ 如何变化？

这个方程就是 continuity equation。

##### 6.2 Continuity Equation 的公式

如果：

$$
X_0 \sim p_{\mathrm{init}}
$$

并且：

$$
\frac{d}{dt}X_t = u_t(X_t)
$$

那么 $X_t$ 的分布 $p_t$ 满足：

$$
\frac{d}{dt}p_t(x)=
-\operatorname{div}(p_t u_t)(x)
$$

其中：

- $\operatorname{div}$ 是散度；
- $p_tu_t$ 可以理解为 probability mass 的流量；
- 右边描述了概率质量从点 $x$ 附近流入或流出的情况。

##### 6.3 直观解释

Continuity equation 可以理解为概率质量守恒：

某处概率密度的变化=流入的概率质量−流出的概率质量

如果一个区域流入的概率质量比流出的多，那么这个区域的概率密度会上升。

如果流出的概率质量比流入的多，那么这个区域的概率密度会下降。

因此：

$$
-\operatorname{div}(p_tu_t)(x)
$$

就是在描述由于向量场搬运概率质量而造成的密度变化。

8.4 它为什么重要？

Continuity equation 是连接下面两个层面的桥梁：

单个样本层面：

$$
\frac{d}{dt}X_t = u_t(X_t)
$$

整体分布层面：

$$
X_t \sim p_t
$$
	​
它告诉我们：

> 如果想让 ODE 的样本分布沿着某条 probability path $p_t$ 走，那么向量场 $u_t$ 必须让 $p_t$ 满足 continuity equation。

这也是 marginal vector field 公式成立的数学基础。

#### 7. Conditional Score Function

##### 7.1 Conditional Score Function 的定义

前面我们定义了 conditional probability path：

$$
p_t(x \mid z)
$$

它表示：给定某一个数据点 $z$ 时，时间 $t$ 的中间样本 $x$ 的概率分布。

在这个条件分布上，我们可以定义 **conditional score function**：

$$
\nabla_x \log p_t(x \mid z)
$$

它表示：

> 在给定数据点 $z$ 的情况下，时间 $t$ 时，当前位置 $x$ 的概率密度上升最快的方向。

这里的梯度是对 $x$ 求的，所以更严格地写是：

$$
\nabla_x \log p_t(x \mid z)
$$

但有时候为了简洁，也会写成：

$$
\nabla \log p_t(x \mid z)
$$

##### 7.2 score function 的直观理解

score function 是 log probability density 的梯度：

$$
\nabla_x \log p_t(x \mid z)
$$

可以把概率密度想象成一张地形图：

- 概率密度高的地方像山峰；
- 概率密度低的地方像山谷；
- score function 指向“往山上走最快”的方向。

所以：

$$
\nabla_x \log p_t(x \mid z)
$$

告诉我们：

> 在 conditional distribution $p_t(x \mid z)$ 中，如果当前点是 $x$，应该往哪个方向移动，概率密度会增加得最快。

对于生成模型来说，这个方向很重要，因为它可以告诉样本如何从低概率区域移动到高概率区域。

##### 7.3 Gaussian Path 下的 Conditional Score

对于 Gaussian conditional probability path：

$$
p_t(x \mid z)=
\mathcal{N}(\alpha_t z,\beta_t^2I_d)
$$

它的均值是：

$$
\alpha_t z
$$

协方差是：

$$
\beta_t^2I_d
$$

因此它的 log density 中和 $x$ 有关的部分是：

$$
-\frac{1}{2\beta_t^2}
\|x-\alpha_t z\|^2
$$

对 $x$ 求梯度：

$$
\nabla_x \log p_t(x \mid z)=
-\frac{x-\alpha_t z}{\beta_t^2}
$$

这就是 Gaussian path 下的 conditional score function。

##### 7.4 这个公式怎么理解？

公式：

$$
\nabla_x \log p_t(x \mid z)
=
-\frac{x-\alpha_t z}{\beta_t^2}
$$

可以这样理解：

- $\alpha_t z$ 是当前 conditional Gaussian 的中心；
- $x-\alpha_t z$ 表示当前点 $x$ 偏离中心的方向；
- 前面的负号表示把 $x$ 往中心 $\alpha_t z$ 拉回去；
- $\beta_t^2$ 是方差，控制拉回去的强度。

如果 $x$ 离中心很远，那么 score 的大小较大，说明应该更强地往中心移动。

如果 $x$ 已经很接近中心，那么 score 接近 0，说明它已经在高概率区域附近。

所以 Gaussian conditional score 的作用是：

> 把当前点 $x$ 往 conditional distribution 的中心 $\alpha_t z$ 拉回去。

##### 7.5 和 Conditional Vector Field 的区别

Conditional vector field 是：

$$
u_t^{\mathrm{target}}(x \mid z)
$$

它描述的是：

> 样本在时间 $t$、位置 $x$ 时，应该以什么速度运动，才能让分布沿着 conditional probability path 演化。

Conditional score function 是：

$$
\nabla_x \log p_t(x \mid z)
$$

它描述的是：

> 在 conditional distribution 中，当前位置 $x$ 的概率密度往哪个方向增加最快。

所以二者的关注点不同：

```text
conditional vector field：描述样本如何随时间运动

conditional score function：描述概率密度往哪里变大
```

#### 8. Marginal Score Function

$$
\nabla_x \log p_t(x)=
\int
\nabla_x \log p_t(x \mid z)
\frac{
p_t(x \mid z)p_{\mathrm{data}}(z)
}{
p_t(x)
}
dz
$$

含义：

> 对所有可能数据点 $z$ 的 conditional score 做后验加权平均，得到整体分布 $p_t$ 的 score。

也可以写成：

$$
\nabla_x \log p_t(x)=
\mathbb{E}
\left[
\nabla_x \log p_t(x \mid z)
\mid x
\right]
$$

它是 Diffusion Model 真正需要学习的对象。

理解：

- conditional score：已知 $z$ 时，哪里概率更高；
- marginal score：不知道具体 $z$ 时，整体分布中哪里概率更高。
