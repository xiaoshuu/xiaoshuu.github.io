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
\nabla_x \log p_t(x \mid z)=
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

#### 9. SDE 与 Fokker–Planck Equation

##### 9.1 从 ODE 到 SDE

前面 Flow Model 使用的是 ODE：

$$
dX_t = u_t(X_t)dt
$$

这个式子表示：

> 样本 $X_t$ 按照向量场 $u_t$ 确定性地运动。

也就是说，如果初始点 $X_0$ 已经确定，那么后面的整条轨迹也是确定的。

Diffusion Model 使用的是 SDE：

$$
dX_t = u_t(X_t)dt + \sigma_t dW_t
$$

它比 ODE 多了一项随机噪声：

$$
\sigma_t dW_t
$$

其中：

- $u_t(X_t)dt$ 是确定性的运动项，也叫 drift term；
- $\sigma_t dW_t$ 是随机扩散项，也叫 diffusion term；
- $\sigma_t$ 是 diffusion coefficient，用来控制随机噪声的强度；
- $W_t$ 是 Brownian motion，也叫 Wiener process。

所以可以这样理解：

**ODE 描述确定性运动。**

**SDE 描述确定性运动加随机扰动。**

如果：

$$
\sigma_t = 0
$$

那么随机项消失，SDE 就退化成 ODE：

$$
dX_t = u_t(X_t)dt
$$

##### 9.2 为什么需要 Fokker–Planck Equation？

ODE 和 SDE 描述的是单个样本怎么运动。

例如：

$$
dX_t = u_t(X_t)dt + \sigma_t dW_t
$$

这个式子描述的是某一个样本 $X_t$ 的运动过程。

但是生成模型真正关心的不是单个样本，而是整个样本分布。

也就是说，我们真正关心的是：

$$
X_t \sim p_t
$$

其中，$p_t$ 表示时间 $t$ 时所有样本形成的概率分布。

因此我们需要一个方程来描述：

> 如果每个样本都按照 SDE 运动，那么整体概率密度 $p_t(x)$ 会如何变化？

这个描述 SDE 下概率密度演化的方程，就是 **Fokker–Planck Equation**。

##### 9.3 先回顾 ODE 对应的 Continuity Equation

对于 ODE：

$$
dX_t = u_t(X_t)dt
$$

也可以写作：

$$
\frac{d}{dt}X_t = u_t(X_t)
$$

如果：

$$
X_t \sim p_t
$$

那么分布 $p_t(x)$ 的变化满足 continuity equation：

$$
\frac{\partial}{\partial t}p_t(x)
=
-\operatorname{div}\left(p_tu_t\right)(x)
$$

这个式子的直观含义是：

> 概率密度的变化来自概率质量的流动。

其中：

$$
p_t(x)u_t(x)
$$

可以理解为概率质量的流量。

如果某个区域流入的概率质量多，流出的概率质量少，那么这个区域的概率密度会上升。

如果某个区域流出的概率质量多，流入的概率质量少，那么这个区域的概率密度会下降。

因此，continuity equation 描述的是：

> 在确定性向量场 $u_t$ 搬运下，概率分布 $p_t$ 如何变化。

##### 9.4 SDE 对应的 Fokker–Planck Equation

对于 SDE：

$$
dX_t = u_t(X_t)dt + \sigma_t dW_t
$$

如果：

$$
X_t \sim p_t
$$

那么概率密度 $p_t(x)$ 满足：

$$
\frac{\partial}{\partial t}p_t(x)
=
-\operatorname{div}\left(p_tu_t\right)(x)
+
\frac{\sigma_t^2}{2}\Delta p_t(x)
$$

这就是 **Fokker–Planck Equation**。

它比 continuity equation 多了一项：

$$
\frac{\sigma_t^2}{2}\Delta p_t(x)
$$

因为 SDE 比 ODE 多了随机噪声：

$$
\sigma_t dW_t
$$

所以 Fokker–Planck Equation 可以理解为：

> SDE 下的分布变化 = 确定性流动造成的变化 + 随机扩散造成的变化。

##### 9.5 公式中每一项的含义

Fokker–Planck Equation 是：

$$
\frac{\partial}{\partial t}p_t(x)
=
-\operatorname{div}\left(p_tu_t\right)(x)
+
\frac{\sigma_t^2}{2}\Delta p_t(x)
$$

第一项：

$$
-\operatorname{div}\left(p_tu_t\right)(x)
$$

表示确定性向量场 $u_t$ 对概率质量的搬运。

这一项和 ODE 的 continuity equation 完全一样。

它描述的是：

> 概率质量沿着向量场 $u_t$ 流动。

第二项：

$$
\frac{\sigma_t^2}{2}\Delta p_t(x)
$$

表示随机噪声造成的扩散。

其中：

- $\Delta$ 是 Laplacian，可以理解为空间上的二阶导数；
- $\sigma_t^2$ 控制扩散强度；
- $\sigma_t$ 越大，随机扩散越强。

所以：

$$
-\operatorname{div}\left(p_tu_t\right)(x)
$$

对应 transport，也就是“流动”。

而：

$$
\frac{\sigma_t^2}{2}\Delta p_t(x)
$$

对应 diffusion，也就是“扩散”。

##### 9.6 Laplacian $\Delta p_t(x)$ 直观上是什么？

在一维情况下，Laplacian 就是二阶导数：

$$
\Delta p_t(x)
=
\frac{\partial^2}{\partial x^2}p_t(x)
$$

在高维情况下：

$$
\Delta p_t(x)
=
\sum_{i=1}^d
\frac{\partial^2}{\partial x_i^2}p_t(x)
$$

直观上，Laplacian 描述的是一个函数在空间中如何弯曲。

在 Fokker–Planck Equation 中：

$$
\Delta p_t(x)
$$

描述的是概率密度如何因为随机噪声而向周围扩散。

随机噪声会让概率质量从高密度区域向周围低密度区域扩散，所以这一项和 heat equation 很像。

这也是为什么有时候会说：

> Fokker–Planck Equation = continuity equation + heat equation。

其中：

- continuity equation 部分来自 drift；
- heat equation 部分来自 diffusion。

##### 9.7 当 $\sigma_t = 0$ 时会发生什么？

如果：

$$
\sigma_t = 0
$$

那么 SDE 变成 ODE：

$$
dX_t = u_t(X_t)dt
$$

同时 Fokker–Planck Equation 中的扩散项消失：

$$
\frac{\sigma_t^2}{2}\Delta p_t(x) = 0
$$

于是 Fokker–Planck Equation 退化为：

$$
\frac{\partial}{\partial t}p_t(x)
=
-\operatorname{div}\left(p_tu_t\right)(x)
$$

这正是 ODE 对应的 continuity equation。

所以可以记住：

> Continuity equation 是 ODE 情况下的分布演化方程。

> Fokker–Planck Equation 是 SDE 情况下的分布演化方程。

##### 9.8 为什么这一节对生成模型重要？

生成模型的目标不是只生成一个样本，而是让整体分布从初始分布变成数据分布：

$$
p_{\mathrm{init}}
\longrightarrow
p_{\mathrm{data}}
$$

也就是说，我们关心的是：

$$
X_0 \sim p_{\mathrm{init}}
$$

经过 ODE 或 SDE 演化后：

$$
X_1 \sim p_{\mathrm{data}}
$$

因此，我们必须理解：

> 当样本按照某个 ODE 或 SDE 运动时，它们的整体分布 $p_t$ 会如何变化。

对于 Flow Model：

$$
dX_t = u_t(X_t)dt
$$

它的分布演化由 continuity equation 描述：

$$
\frac{\partial}{\partial t}p_t(x)
=
-\operatorname{div}\left(p_tu_t\right)(x)
$$

对于 Diffusion Model：

$$
dX_t = u_t(X_t)dt + \sigma_t dW_t
$$

它的分布演化由 Fokker–Planck Equation 描述：

$$
\frac{\partial}{\partial t}p_t(x)
=
-\operatorname{div}\left(p_tu_t\right)(x)
+
\frac{\sigma_t^2}{2}\Delta p_t(x)
$$

所以这一节的核心作用是：

> 把“单个样本的随机运动”与“整体概率分布的演化”联系起来。

##### 9.9 和 SDE Extension Trick 的关系

后面会讲到 SDE extension trick。

它会构造这样一个 SDE：

$$
dX_t =
\left[
u_t^{\mathrm{target}}(X_t)
+
\frac{\sigma_t^2}{2}
\nabla_x \log p_t(X_t)
\right]dt
+
\sigma_t dW_t
$$

这个式子看起来比普通 SDE 多了一个 score correction：

$$
\frac{\sigma_t^2}{2}
\nabla_x \log p_t(X_t)
$$

为什么需要这一项？

原因就来自 Fokker–Planck Equation。

随机噪声项：

$$
\sigma_t dW_t
$$

会在 Fokker–Planck Equation 中产生扩散项：

$$
\frac{\sigma_t^2}{2}\Delta p_t(x)
$$

如果不加修正，分布会因为随机噪声而跑偏。

而 score correction 会在 Fokker–Planck Equation 中产生一个相反的项，正好抵消这个扩散项。

这样一来，即使加入随机噪声，整体分布仍然可以保持在目标 probability path：

$$
X_t \sim p_t
$$

所以 Fokker–Planck Equation 是理解 SDE extension trick 的关键。

##### 9.10 小结

Fokker–Planck Equation 描述的是 SDE 下概率密度的演化。

对于 SDE：

$$
dX_t = u_t(X_t)dt + \sigma_t dW_t
$$

如果：

$$
X_t \sim p_t
$$

那么：

$$
\frac{\partial}{\partial t}p_t(x)
=
-\operatorname{div}\left(p_tu_t\right)(x)
+
\frac{\sigma_t^2}{2}\Delta p_t(x)
$$

其中：

- $-\operatorname{div}(p_tu_t)(x)$ 表示确定性向量场造成的概率质量流动；
- $\frac{\sigma_t^2}{2}\Delta p_t(x)$ 表示随机噪声造成的概率扩散。

如果 $\sigma_t=0$，Fokker–Planck Equation 就退化成 continuity equation。

一句话总结：

> Fokker–Planck Equation 是描述 SDE 如何改变整体概率分布的方程，它把 drift 引起的流动和 diffusion 引起的扩散统一到一个公式里。

#### 10 SDE Extension Trick

##### 10.1 这一页在讲什么？

前面我们已经知道，Flow Model 的 ODE 是：

$$
dX_t = u_t^{\mathrm{target}}(X_t)dt
$$

如果：

$$
X_0 \sim p_0
$$

并且 $u_t^{\mathrm{target}}$ 是正确的 marginal vector field，那么：

$$
X_t \sim p_t
$$

也就是说，ODE 会让样本分布沿着我们构造好的 probability path $p_t$ 演化。

这一页要说明的是：

> 我们不仅可以用 ODE 产生这条 probability path，也可以用一类 SDE 产生同一条 probability path。

也就是说，除了确定性的 Flow ODE：

$$
dX_t = u_t^{\mathrm{target}}(X_t)dt
$$

我们还可以构造带随机噪声的 SDE：

$$
dX_t =
\left[
u_t^{\mathrm{target}}(X_t)
+
\frac{\sigma_t^2}{2}
\nabla_x \log p_t(X_t)
\right]dt
+
\sigma_t dW_t
$$

并且仍然有：

$$
X_t \sim p_t
$$

这就是 **SDE extension trick**。

##### 10.2 公式

假设 $u_t^{\mathrm{target}}(x)$ 是前面构造出来的 marginal vector field。

也就是说，ODE：

$$
dX_t = u_t^{\mathrm{target}}(X_t)dt
$$

会让样本分布满足：

$$
X_t \sim p_t
$$

那么对于任意 diffusion coefficient：

$$
\sigma_t \geq 0
$$

我们可以构造下面这个 SDE：

$$
dX_t =
\left[
u_t^{\mathrm{target}}(X_t)
+
\frac{\sigma_t^2}{2}
\nabla_x \log p_t(X_t)
\right]dt
+
\sigma_t dW_t
$$

只要初始分布满足：

$$
X_0 \sim p_0
$$

那么这个 SDE 的边缘分布仍然满足：

$$
X_t \sim p_t,
\qquad 0 \leq t \leq 1
$$

##### 10.3 每一项是什么意思？

这个 SDE 中一共有三部分。

第一部分是：

$$
u_t^{\mathrm{target}}(X_t)
$$

这是原本 Flow Model 的 marginal vector field。

它负责把样本整体从初始分布 $p_0$ 推向数据分布 $p_1$。

第二部分是：

$$
\sigma_t dW_t
$$

这是随机扩散项。

它会给样本运动加入随机扰动。

其中：

- $W_t$ 是 Brownian motion；
- $\sigma_t$ 是 diffusion coefficient；
- $\sigma_t$ 越大，随机扰动越强。

第三部分是：

$$
\frac{\sigma_t^2}{2}
\nabla_x \log p_t(X_t)
$$

这是 score correction，也就是 score 修正项。

其中：

$$
\nabla_x \log p_t(X_t)
$$

是 marginal score function。

它表示在当前分布 $p_t$ 下，概率密度上升最快的方向。

##### 10.4 为什么加入噪声后还要加 score correction？

如果我们只是在原本的 ODE 上直接加噪声：

$$
dX_t =
u_t^{\mathrm{target}}(X_t)dt
+
\sigma_t dW_t
$$

那么随机噪声会让分布额外扩散。

这样一来，$X_t$ 的分布就不一定还是原来的 $p_t$。

也就是说，只加噪声会让样本分布偏离我们设计好的 probability path。

所以需要加入一个修正项：

$$
\frac{\sigma_t^2}{2}
\nabla_x \log p_t(X_t)
$$

这个 score correction 会把样本往当前分布 $p_t$ 的高概率区域拉回去。

因此可以这样理解：

> 随机噪声项 $\sigma_t dW_t$ 会让分布向外扩散，而 score correction 会把样本往高概率区域拉回。两者配合之后，整体分布仍然保持在目标路径 $p_t$ 上。

##### 10.5 为什么这个公式成立？

这一点可以用 Fokker–Planck Equation 验证。

一般的 SDE 写作：

$$
dX_t = b_t(X_t)dt + \sigma_t dW_t
$$

其中 $b_t(x)$ 是 drift。

它对应的 Fokker–Planck Equation 是：

$$
\frac{\partial}{\partial t}p_t(x)
=
-\operatorname{div}(p_t b_t)(x)
+
\frac{\sigma_t^2}{2}
\Delta p_t(x)
$$

现在我们把 drift 设成：

$$
b_t(x)
=
u_t^{\mathrm{target}}(x)
+
\frac{\sigma_t^2}{2}
\nabla_x \log p_t(x)
$$

代入 Fokker–Planck Equation：

$$
\frac{\partial}{\partial t}p_t(x)
=
-\operatorname{div}
\left(
p_t
\left[
u_t^{\mathrm{target}}
+
\frac{\sigma_t^2}{2}
\nabla_x \log p_t
\right]
\right)(x)
+
\frac{\sigma_t^2}{2}
\Delta p_t(x)
$$

展开第一项：

$$
\frac{\partial}{\partial t}p_t(x)
=
-\operatorname{div}
\left(
p_t u_t^{\mathrm{target}}
\right)(x)
-
\frac{\sigma_t^2}{2}
\operatorname{div}
\left(
p_t \nabla_x \log p_t
\right)(x)
+
\frac{\sigma_t^2}{2}
\Delta p_t(x)
$$

注意：

$$
\nabla_x \log p_t(x)
=
\frac{\nabla_x p_t(x)}{p_t(x)}
$$

所以：

$$
p_t(x)\nabla_x \log p_t(x)
=
\nabla_x p_t(x)
$$

因此：

$$
\operatorname{div}
\left(
p_t \nabla_x \log p_t
\right)(x)
=
\operatorname{div}
\left(
\nabla_x p_t
\right)(x)
=
\Delta p_t(x)
$$

代回去：

$$
\frac{\partial}{\partial t}p_t(x)
=
-\operatorname{div}
\left(
p_t u_t^{\mathrm{target}}
\right)(x)
-
\frac{\sigma_t^2}{2}
\Delta p_t(x)
+
\frac{\sigma_t^2}{2}
\Delta p_t(x)
$$

后两项正好抵消：

$$
-
\frac{\sigma_t^2}{2}
\Delta p_t(x)
+
\frac{\sigma_t^2}{2}
\Delta p_t(x)
=
0
$$

所以最后只剩下：

$$
\frac{\partial}{\partial t}p_t(x)
=
-\operatorname{div}
\left(
p_t u_t^{\mathrm{target}}
\right)(x)
$$

这正是原本 ODE 对应的 continuity equation。

也就是说，这个 SDE 和原来的 Flow ODE 会产生同一条 marginal probability path：

$$
X_t \sim p_t
$$

##### 10.6 这个 trick 的核心意义

这个公式说明：

> 同一条 probability path $p_t$，既可以由一个确定性的 ODE 生成，也可以由一族带随机噪声的 SDE 生成。

原来的 ODE 是：

$$
dX_t = u_t^{\mathrm{target}}(X_t)dt
$$

扩展后的 SDE 是：

$$
dX_t =
\left[
u_t^{\mathrm{target}}(X_t)
+
\frac{\sigma_t^2}{2}
\nabla_x \log p_t(X_t)
\right]dt
+
\sigma_t dW_t
$$

其中 $\sigma_t$ 可以控制随机性的大小。

如果：

$$
\sigma_t = 0
$$

那么 SDE 退化成 ODE：

$$
dX_t = u_t^{\mathrm{target}}(X_t)dt
$$

如果：

$$
\sigma_t > 0
$$

那么生成过程会带有随机性，但由于加入了 score correction，边缘分布仍然是 $p_t$。

##### 10.7 为什么这和 Diffusion Model 有关？

Flow Model 主要学习的是 marginal vector field：

$$
u_t^{\mathrm{target}}(x)
$$

Diffusion Model 还需要 score function：

$$
\nabla_x \log p_t(x)
$$

从这个 SDE extension trick 可以看到，score function 出现在 SDE 的 drift 里面：

$$
\frac{\sigma_t^2}{2}
\nabla_x \log p_t(X_t)
$$

也就是说，如果我们想用带噪声的 SDE 来采样，就需要知道：

$$
\nabla_x \log p_t(x)
$$

这就是为什么 diffusion model 要学习 score function。

可以这样理解：

- Flow Model 学 $u_t^{\mathrm{target}}(x)$，用于确定性 ODE 采样；
- Diffusion Model 学 $\nabla_x \log p_t(x)$，用于带随机噪声的 SDE 采样；
- SDE extension trick 把 marginal vector field 和 marginal score function 连接了起来。

##### 10.8 直观总结

原本的 Flow ODE 是：

$$
dX_t = u_t^{\mathrm{target}}(X_t)dt
$$

它的作用是：

> 按照确定性速度场，把分布从 $p_0$ 搬运到 $p_1$。

SDE extension trick 告诉我们，也可以加入随机噪声：

$$
\sigma_t dW_t
$$

但为了不让分布跑偏，必须同时加入 score correction：

$$
\frac{\sigma_t^2}{2}
\nabla_x \log p_t(X_t)
$$

所以最终得到：

$$
dX_t =
\left[
u_t^{\mathrm{target}}(X_t)
+
\frac{\sigma_t^2}{2}
\nabla_x \log p_t(X_t)
\right]dt
+
\sigma_t dW_t
$$

一句话总结：

> SDE extension trick 说明：在 Flow ODE 中加入随机噪声后，只要再加入正确的 score correction，就可以得到一个边缘分布仍然沿着 $p_t$ 演化的 Diffusion SDE。

![20260430165700](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260430165700.png)
![20260430165722](https://cdn.jsdelivr.net/gh/xiaoshuu/img/Picgo/20260430165722.png)

### 如何训练生成式AI

#### 该部分目标

对于 Flow Model，我们希望训练神经网络：

$$
u_t^\theta(x)
$$

让它逼近：

$$
u_t^{\mathrm{target}}(x)
$$

也就是：

$$
u_t^\theta(x)
\approx
u_t^{\mathrm{target}}(x)
$$

对于 Diffusion Model，我们希望训练神经网络：

$$
s_t^\theta(x)
$$

让它逼近：

$$
\nabla_x \log p_t(x)
$$

也就是：

$$
s_t^\theta(x)
\approx
\nabla_x \log p_t(x)
$$

> Flow Matching 训练 $u_t^\theta(x)$。

> Score Matching 训练 $s_t^\theta(x)$。

#### Flow Matching

##### Flow Model 的采样

如果我们已经训练好了 vector field network：

$$
u_t^\theta(x)
$$

那么 Flow Model 的采样过程是：

$$
X_0 \sim p_{\mathrm{init}}
$$

然后解 ODE：

$$
\frac{d}{dt}X_t = u_t^\theta(X_t)
$$

最后返回：

$$
X_1
$$

如果模型训练得好，那么：

$$
X_1 \sim p_{\mathrm{data}}
$$

实际计算时通常用 Euler method：

$$
X_{t+h} = X_t + h u_t^\theta(X_t)
$$

其中：

$$
h = \frac{1}{n}
$$

$n$ 是采样步数。

采样流程可以理解为：

1. 从噪声分布中采样：

$$
X_0 \sim p_{\mathrm{init}}
$$

2. 用神经网络预测当前位置的速度：

$$
u_t^\theta(X_t)
$$

3. 沿着这个速度走一小步：

$$
X_{t+h}=X_t+h u_t^\theta(X_t)
$$

4. 重复直到 $t=1$。

##### 理想的 Flow Matching Loss

我们真正想学的是 marginal vector field：

$$
u_t^{\mathrm{target}}(x)
$$

所以最直接的 loss 是：

$$
\mathcal{L}_{\mathrm{FM}}(\theta) =
\mathbb{E}_{t\sim \mathrm{Unif}[0,1],\ x\sim p_t}
\left[
\left\|
u_t^\theta(x) -
u_t^{\mathrm{target}}(x)
\right\|^2
\right]
$$

这个 loss 的意思是：

> 在任意时间 $t$ 和任意中间点 $x$，让神经网络预测的速度接近真实的 marginal vector field。

但是这个目标有一个问题：

$$
u_t^{\mathrm{target}}(x)
$$

通常不好直接计算。

因为它是 marginal vector field：

$$
u_t^{\mathrm{target}}(x) =
\int
u_t^{\mathrm{target}}(x \mid z)
\frac{
p_t(x \mid z)p_{\mathrm{data}}(z)
}{
p_t(x)
}
dz
$$

里面有对所有数据点 $z$ 的积分。

所以直接训练：

$$
u_t^\theta(x)
\approx
u_t^{\mathrm{target}}(x)
$$

不太现实。

虽然 marginal vector field 不好直接算，但是 conditional vector field 好算。

因此 Flow Matching 的想法是：

> 训练时不用 marginal target，而是使用 conditional target。

也就是训练：

$$
u_t^\theta(x)
\approx
u_t^{\mathrm{target}}(x \mid z)
$$

于是定义训练 loss：

$$
\mathcal{L}_{\mathrm{CFM}}(\theta)=
\mathbb{E}_{z\sim p_{\mathrm{data}},\ t\sim \mathrm{Unif}[0,1],\ x\sim p_t(\cdot\mid z)}
\left[
\left\|
u_t^\theta(x)-
u_t^{\mathrm{target}}(x \mid z)
\right\|^2
\right]
$$

这叫 Conditional Flow Matching。

##### 为什么用 conditional target 也能学到 marginal vector field？

训练时，我们采样：

$$
z \sim p_{\mathrm{data}}
$$

$$
x \sim p_t(\cdot \mid z)
$$

也就是说，训练数据中的 $x$ 是通过某个隐藏的 $z$ 生成出来的。

但是神经网络输入的是：

$$
(x,t)
$$

不是：

$$
(x,t,z)
$$

也就是说，网络不知道当前 $x$ 到底来自哪个 $z$。

当同一个 $x$ 可能来自多个不同的 $z$ 时，网络为了最小化均方误差，最优预测会是这些 conditional vector field 的条件平均：

$$
u_t^\theta(x)=
\mathbb{E}
\left[
u_t^{\mathrm{target}}(x \mid z)
\mid x
\right]
$$

而第二讲已经证明：

$$
\mathbb{E}
\left[
u_t^{\mathrm{target}}(x \mid z)
\mid x
\right]=
u_t^{\mathrm{target}}(x) + C
$$

即

$$
\boxed{
\mathcal{L}_{\mathrm{FM}}
\text{ 和 }
\mathcal{L}_{\mathrm{CFM}}
\text{ 只差一个与 } \theta \text{ 无关的常数}
}
$$

所以，虽然训练时用的是 conditional target：

$$
u_t^{\mathrm{target}}(x \mid z)
$$

但最优情况下，网络学到的是 marginal vector field：

$$
u_t^{\mathrm{target}}(x)
$$

这就是 marginalization trick 在训练中的作用。

##### Flow Matching 训练流程

Flow Matching 的训练算法如下。

给定：

- 数据集：

$$
z \sim p_{\mathrm{data}}
$$

- 神经网络 vector field：

$$
u_t^\theta(x)
$$

每次训练：

1. 从数据集中采样一个真实数据点：

$$
z \sim p_{\mathrm{data}}
$$

2. 采样随机时间：

$$
t \sim \mathrm{Unif}[0,1]
$$

3. 从 conditional probability path 中采样中间点：

$$
x \sim p_t(\cdot \mid z)
$$

4. 计算 conditional vector field：

$$
u_t^{\mathrm{target}}(x \mid z)
$$

5. 计算 loss：

$$
\mathcal{L}(\theta)=
\left\|
u_t^\theta(x)-
u_t^{\mathrm{target}}(x \mid z)
\right\|^2
$$

6. 用梯度下降更新参数 $\theta$。

这就是 Flow Matching 的训练过程。

##### Gaussian Probability Path 下的 Flow Matching

常用的 Gaussian probability path 是：

$$
p_t(x \mid z)=
\mathcal{N}(\alpha_t z,\beta_t^2 I_d)
$$

等价采样形式是：

$$
x_t = \alpha_t z + \beta_t \epsilon
$$

其中：

$$
\epsilon \sim \mathcal{N}(0,I_d)
$$

对于这条 path，conditional vector field 是：

$$
u_t^{\mathrm{target}}(x \mid z) =
\left(
\dot{\alpha}_t -
\frac{\dot{\beta}_t}{\beta_t}\alpha_t
\right)z +
\frac{\dot{\beta}_t}{\beta_t}x
$$

所以 Gaussian path 下的 Flow Matching loss 是：

$$
\mathcal{L}_{\mathrm{CFM}}(\theta) =
\mathbb{E}_{t,z,\epsilon}
\left[
\left\|
u_t^\theta(\alpha_t z+\beta_t\epsilon) -
u_t^{\mathrm{target}}(\alpha_t z+\beta_t\epsilon \mid z)
\right\|^2
\right]
$$

##### Straight Line Path 的特殊情况

一个非常直观的选择是 straight line path：

$$
\alpha_t = t
$$

$$
\beta_t = 1-t
$$

此时：

$$
x_t = tz + (1-t)\epsilon
$$

它表示从噪声 $\epsilon$ 到数据点 $z$ 的直线插值。

当 $t=0$：

$$
x_0 = \epsilon
$$

当 $t=1$：

$$
x_1 = z
$$

对时间求导：

$$
\frac{d}{dt}x_t
=
z-\epsilon
$$

所以 conditional vector field 可以写成：

$$
u_t^{\mathrm{target}}(x_t \mid z)
=
z-\epsilon
$$

也可以把 $\epsilon$ 消掉。

由：

$$
x = tz+(1-t)\epsilon
$$

得到：

$$
\epsilon
=
\frac{x-tz}{1-t}
$$

代入：

$$
z-\epsilon
=
\frac{z-x}{1-t}
$$

所以：

$$
u_t^{\mathrm{target}}(x \mid z)
=
\frac{z-x}{1-t}
$$

这个公式的直观意义是：

> 当前在 $x$，目标是 $z$，剩余时间是 $1-t$，所以速度应该是目标位置减当前位置，再除以剩余时间。

也就是：

$$
\text{速度}
=
\frac{\text{目标位置}-\text{当前位置}}{\text{剩余时间}}
$$

---

##### 10. Flow Matching 训练完成后怎么生成？

训练完成后，我们得到了：

$$
u_t^\theta(x)
$$

然后生成时不再需要数据点 $z$。

采样过程是：

1. 从初始分布采样：

$$
X_0 \sim p_{\mathrm{init}}
$$

2. 解 ODE：

$$
\frac{d}{dt}X_t
=
u_t^\theta(X_t)
$$

3. 从 $t=0$ 积分到 $t=1$。

4. 返回：

$$
X_1
$$

如果训练得好：

$$
X_1 \sim p_{\mathrm{data}}
$$

所以 Flow Matching 的完整逻辑是：

> 训练时用数据点 $z$ 构造 conditional target；生成时只使用学到的 marginal vector field，从噪声走到数据。

---

# Part 2：Score Matching

##### 11. Diffusion Model 为什么需要 Score？

第二讲里我们知道，Diffusion Model 需要 marginal score function：

$$
\nabla_x \log p_t(x)
$$

它表示当前分布 $p_t$ 中概率密度增加最快的方向。

从 SDE extension trick 可以看到，如果要用带噪声的 SDE 采样，需要：

$$
dX_t =
\left[
u_t^{\mathrm{target}}(X_t)
+
\frac{\sigma_t^2}{2}
\nabla_x \log p_t(X_t)
\right]dt
+
\sigma_t dW_t
$$

这里的：

$$
\nabla_x \log p_t(X_t)
$$

就是 score function。

因此，Diffusion Model 需要训练一个 score network：

$$
s_t^\theta(x)
$$

让它逼近：

$$
\nabla_x \log p_t(x)
$$

也就是：

$$
s_t^\theta(x)
\approx
\nabla_x \log p_t(x)
$$

---

##### 12. 理想的 Score Matching Loss

最直接的训练目标是：

$$
\mathcal{L}_{\mathrm{SM}}(\theta)
=
\mathbb{E}_{t\sim \mathrm{Unif}[0,1],\ x\sim p_t}
\left[
\left\|
s_t^\theta(x)
-
\nabla_x \log p_t(x)
\right\|^2
\right]
$$

这个 loss 的意思是：

> 在任意时间 $t$ 和任意中间点 $x$，让神经网络预测的 score 接近真实的 marginal score。

但是问题是：

$$
\nabla_x \log p_t(x)
$$

不好直接算。

因为：

$$
p_t(x)
=
\int p_t(x \mid z)p_{\mathrm{data}}(z)dz
$$

所以 marginal score 也涉及对所有数据点 $z$ 的积分。

---

##### 13. Score Matching 的关键想法

虽然 marginal score 不好算，但是 conditional score 好算。

也就是说：

$$
\nabla_x \log p_t(x)
$$

不好算。

但是：

$$
\nabla_x \log p_t(x \mid z)
$$

通常可以算。

所以 Score Matching 的训练目标是：

$$
\mathcal{L}_{\mathrm{DSM}}(\theta)
=
\mathbb{E}_{z,t,x}
\left[
\left\|
s_t^\theta(x)
-
\nabla_x \log p_t(x \mid z)
\right\|^2
\right]
$$

其中：

$$
z \sim p_{\mathrm{data}}
$$

$$
t \sim \mathrm{Unif}[0,1]
$$

$$
x \sim p_t(\cdot \mid z)
$$

这就是 Denoising Score Matching。

它和 Flow Matching 的结构非常像。

Flow Matching 是：

$$
u_t^\theta(x)
\approx
u_t^{\mathrm{target}}(x \mid z)
$$

Score Matching 是：

$$
s_t^\theta(x)
\approx
\nabla_x \log p_t(x \mid z)
$$

---

##### 14. 为什么用 conditional score 能学到 marginal score？

原因和 Flow Matching 一样。

训练时，神经网络看到的是：

$$
(x,t)
$$

但它不知道当前 $x$ 来自哪个 $z$。

对于固定的 $x$ 和 $t$，最小化均方误差的最优预测是 conditional score 的条件期望：

$$
s_t^\theta(x)
=
\mathbb{E}
\left[
\nabla_x \log p_t(x \mid z)
\mid x
\right]
$$

而第二讲已经得到：

$$
\mathbb{E}
\left[
\nabla_x \log p_t(x \mid z)
\mid x
\right]
=
\nabla_x \log p_t(x)
$$

所以最优情况下：

$$
s_t^\theta(x)
=
\nabla_x \log p_t(x)
$$

也就是说：

> 虽然训练时使用 conditional score，但网络最终学到的是 marginal score。

---

##### 15. Score Matching 训练流程

给定：

- 数据集：

$$
z \sim p_{\mathrm{data}}
$$

- score network：

$$
s_t^\theta(x)
$$

每次训练：

1. 从数据集中采样真实数据点：

$$
z \sim p_{\mathrm{data}}
$$

2. 采样随机时间：

$$
t \sim \mathrm{Unif}[0,1]
$$

3. 从 conditional probability path 中采样中间点：

$$
x \sim p_t(\cdot \mid z)
$$

4. 计算 conditional score：

$$
\nabla_x \log p_t(x \mid z)
$$

5. 计算 loss：

$$
\mathcal{L}(\theta)
=
\left\|
s_t^\theta(x)
-
\nabla_x \log p_t(x \mid z)
\right\|^2
$$

6. 用梯度下降更新参数 $\theta$。

---

##### 16. Gaussian Path 下的 Conditional Score

对于 Gaussian probability path：

$$
p_t(x \mid z)
=
\mathcal{N}(\alpha_t z,\beta_t^2 I_d)
$$

conditional score 有显式公式：

$$
\nabla_x \log p_t(x \mid z)
=
-\frac{x-\alpha_t z}{\beta_t^2}
$$

如果：

$$
x = \alpha_t z + \beta_t \epsilon
$$

其中：

$$
\epsilon \sim \mathcal{N}(0,I_d)
$$

那么：

$$
x-\alpha_t z
=
\beta_t \epsilon
$$

所以：

$$
\nabla_x \log p_t(x \mid z)
=
-\frac{\beta_t \epsilon}{\beta_t^2}
=
-\frac{\epsilon}{\beta_t}
$$

因此 Gaussian path 下的 denoising score matching loss 是：

$$
\mathcal{L}_{\mathrm{DSM}}(\theta)
=
\mathbb{E}_{t,z,\epsilon}
\left[
\left\|
s_t^\theta(\alpha_t z+\beta_t\epsilon)
+
\frac{\epsilon}{\beta_t}
\right\|^2
\right]
$$

这里是加号，因为目标是：

$$
s_t^\theta(x)
\approx
-\frac{\epsilon}{\beta_t}
$$

所以：

$$
s_t^\theta(x)
+
\frac{\epsilon}{\beta_t}
\approx
0
$$

---

##### 17. 为什么叫 Denoising Score Matching？

因为训练时输入的：

$$
x_t = \alpha_t z + \beta_t \epsilon
$$

是一个被噪声污染的版本。

其中：

- $z$ 是干净数据；
- $\epsilon$ 是噪声；
- $\beta_t$ 控制噪声强度。

score network 要学习：

$$
s_t^\theta(x_t)
\approx
-\frac{\epsilon}{\beta_t}
$$

也就是说，它要从 noisy sample 中推断噪声方向，或者推断如何往更高概率区域移动。

所以这个过程可以理解为：

> 给模型一个带噪声的样本，让它学习如何去噪。

这就是 denoising 的含义。

---

##### 18. Noise Predictor 的写法

很多 diffusion model 不直接预测 score：

$$
s_t^\theta(x)
$$

而是预测噪声：

$$
\epsilon_t^\theta(x)
$$

因为在 Gaussian path 下：

$$
s_t^\theta(x)
\approx
-\frac{\epsilon}{\beta_t}
$$

所以可以定义：

$$
\epsilon_t^\theta(x)
=
-\beta_t s_t^\theta(x)
$$

这样训练目标可以改写成预测噪声：

$$
\epsilon_t^\theta(x_t)
\approx
\epsilon
$$

对应的 loss 是：

$$
\mathcal{L}(\theta)
=
\left\|
\epsilon_t^\theta(x_t)
-
\epsilon
\right\|^2
$$

这也是很多实际 diffusion model 训练时常见的形式。

---

##### 19. 低 $\beta_t$ 时的数值不稳定

在 Gaussian path 下，score target 是：

$$
-\frac{\epsilon}{\beta_t}
$$

如果：

$$
\beta_t
$$

很小，那么：

$$
\frac{\epsilon}{\beta_t}
$$

会变得很大。

这会导致训练数值不稳定。

尤其在 $t$ 接近 1 时，通常：

$$
\beta_t \rightarrow 0
$$

所以这个 target 可能会爆炸。

这也是为什么实际 diffusion model 里经常会使用不同的参数化方式，例如预测噪声、预测 clean data，或者对 loss 加权。

---

# Part 3：Stochastic Sampling with Diffusion Models

##### 20. SDE Extension Trick 回顾

第二讲里已经讲过 SDE extension trick。

如果：

$$
u_t^{\mathrm{target}}(x)
$$

是正确的 marginal vector field，并且：

$$
\nabla_x \log p_t(x)
$$

是正确的 marginal score function，那么对于任意 diffusion coefficient：

$$
\sigma_t \geq 0
$$

下面这个 SDE 仍然会让样本分布沿着 $p_t$ 走：

$$
dX_t =
\left[
u_t^{\mathrm{target}}(X_t)
+
\frac{\sigma_t^2}{2}
\nabla_x \log p_t(X_t)
\right]dt
+
\sigma_t dW_t
$$

也就是说：

$$
X_t \sim p_t
$$

---

##### 21. 训练后如何用神经网络采样？

训练完成后，我们用神经网络替换真实 target。

用：

$$
u_t^\theta(x)
$$

近似：

$$
u_t^{\mathrm{target}}(x)
$$

用：

$$
s_t^\theta(x)
$$

近似：

$$
\nabla_x \log p_t(x)
$$

于是 SDE 采样公式变成：

$$
dX_t =
\left[
u_t^\theta(X_t)
+
\frac{\sigma_t^2}{2}
s_t^\theta(X_t)
\right]dt
+
\sigma_t dW_t
$$

从：

$$
X_0 \sim p_{\mathrm{init}}
$$

开始，模拟这个 SDE 到 $t=1$，得到：

$$
X_1
$$

如果训练得好，那么：

$$
X_1 \sim p_{\mathrm{data}}
$$

---

##### 22. Euler-Maruyama 采样

对于 SDE：

$$
dX_t =
\left[
u_t^\theta(X_t)
+
\frac{\sigma_t^2}{2}
s_t^\theta(X_t)
\right]dt
+
\sigma_t dW_t
$$

可以用 Euler-Maruyama method 离散化。

设步长为：

$$
h = \frac{1}{n}
$$

则更新公式是：

$$
X_{t+h}
=
X_t
+
h
\left[
u_t^\theta(X_t)
+
\frac{\sigma_t^2}{2}
s_t^\theta(X_t)
\right]
+
\sigma_t \sqrt{h}\xi
$$

其中：

$$
\xi \sim \mathcal{N}(0,I_d)
$$

这里比 ODE 的 Euler method 多了一项随机噪声：

$$
\sigma_t \sqrt{h}\xi
$$

如果：

$$
\sigma_t=0
$$

那么就退化成 Flow Model 的 ODE 采样：

$$
X_{t+h}
=
X_t
+
h u_t^\theta(X_t)
$$

---

##### 23. Diffusion coefficient 的作用

在 SDE 采样公式中：

$$
dX_t =
\left[
u_t^\theta(X_t)
+
\frac{\sigma_t^2}{2}
s_t^\theta(X_t)
\right]dt
+
\sigma_t dW_t
$$

其中：

$$
\sigma_t
$$

是 diffusion coefficient。

它控制随机性的大小。

如果：

$$
\sigma_t=0
$$

那么采样是确定性的，等价于 Flow ODE。

如果：

$$
\sigma_t>0
$$

那么采样是随机的。

此时：

- $\sigma_t dW_t$ 会让样本随机扩散；
- $\frac{\sigma_t^2}{2}s_t^\theta(X_t)$ 会把样本往高概率区域拉回。

所以可以这样理解：

> diffusion coefficient 控制“扩散得有多厉害”，score term 负责“把扩散后的样本拉回正确分布”。

---

# Part 4：Denoising Diffusion Models

##### 24. DDM 是什么？

课程中说的 DDM 指的是 Denoising Diffusion Model。

很多人会直接把 Denoising Diffusion Model 简称为 Diffusion Model。

在这门课的框架里：

> Denoising Diffusion Model = 使用 Gaussian probability path 的 Diffusion Model。

也就是使用：

$$
p_t(x \mid z)
=
\mathcal{N}(\alpha_t z,\beta_t^2I_d)
$$

作为 conditional probability path。

这是课程里一直使用的标准例子。

---

##### 25. DDM 的特殊性质：Score for Free

对于 Gaussian probability path，我们有两个公式。

第一个是 conditional vector field：

$$
u_t^{\mathrm{target}}(x \mid z)
=
\left(
\dot{\alpha}_t
-
\frac{\dot{\beta}_t}{\beta_t}\alpha_t
\right)z
+
\frac{\dot{\beta}_t}{\beta_t}x
$$

第二个是 conditional score：

$$
\nabla_x \log p_t(x \mid z)
=
-\frac{x-\alpha_t z}{\beta_t^2}
$$

这两个公式都有一个共同点：

> 它们都是 $x$ 和 $z$ 的线性组合。

因此，在 Gaussian path 下，conditional vector field 和 conditional score 可以互相转换。

这就是 slides 里说的：

> Score for free.

意思是：

> 对 Denoising Diffusion Models 来说，不一定需要分别训练 vector field network 和 score network，因为二者可以通过公式互相转换。

---

##### 26. Conditional 版本的转换公式

对于 Gaussian path，可以推导出：

$$
u_t^{\mathrm{target}}(x \mid z)
=
\left(
\beta_t^2\frac{\dot{\alpha}_t}{\alpha_t}
-
\dot{\beta}_t\beta_t
\right)
\nabla_x \log p_t(x \mid z)
+
\frac{\dot{\alpha}_t}{\alpha_t}x
$$

这个公式表示：

> 如果知道 conditional score，就可以得到 conditional vector field。

注意，这个公式里有：

$$
\frac{\dot{\alpha}_t}{\alpha_t}
$$

所以在 $\alpha_t=0$ 的端点附近需要小心处理。

---

##### 27. Marginal 版本的转换公式

对所有 $z$ 做 marginalization 后，可以得到 marginal 版本：

$$
u_t^{\mathrm{target}}(x)
=
\left(
\beta_t^2\frac{\dot{\alpha}_t}{\alpha_t}
-
\dot{\beta}_t\beta_t
\right)
\nabla_x \log p_t(x)
+
\frac{\dot{\alpha}_t}{\alpha_t}x
$$

也就是说：

> marginal vector field 可以由 marginal score function 转换得到。

如果我们训练的是 score network：

$$
s_t^\theta(x)
\approx
\nabla_x \log p_t(x)
$$

那么可以得到一个 vector field network：

$$
u_t^\theta(x)
=
\left(
\beta_t^2\frac{\dot{\alpha}_t}{\alpha_t}
-
\dot{\beta}_t\beta_t
\right)
s_t^\theta(x)
+
\frac{\dot{\alpha}_t}{\alpha_t}x
$$

这说明：

> 在 DDM 中，训练 score network 后，也可以后处理得到 vector field network。

---

##### 28. 为什么早期 diffusion model 只做 score matching？

因为在 DDM 中，score 和 vector field 可以互相转换。

所以早期 diffusion model 主要训练 score network：

$$
s_t^\theta(x)
$$

也就是做 Score Matching。

训练完 score network 之后，可以用 score 来构造 SDE 采样，也可以通过转换公式得到对应的 vector field。

因此课程里说：

> 第一代 diffusion models 只做 score matching。

---

# Part 5：完整算法总结

##### 29. Flow Matching 完整流程

训练阶段：

1. 采样数据：

$$
z \sim p_{\mathrm{data}}
$$

2. 采样时间：

$$
t \sim \mathrm{Unif}[0,1]
$$

3. 采样中间点：

$$
x \sim p_t(\cdot \mid z)
$$

4. 计算 conditional vector field：

$$
u_t^{\mathrm{target}}(x \mid z)
$$

5. 训练：

$$
\mathcal{L}(\theta)
=
\left\|
u_t^\theta(x)
-
u_t^{\mathrm{target}}(x \mid z)
\right\|^2
$$

采样阶段：

1. 采样噪声：

$$
X_0 \sim p_{\mathrm{init}}
$$

2. 解 ODE：

$$
dX_t = u_t^\theta(X_t)dt
$$

3. 返回：

$$
X_1
$$

---

##### 30. Score Matching 完整流程

训练阶段：

1. 采样数据：

$$
z \sim p_{\mathrm{data}}
$$

2. 采样时间：

$$
t \sim \mathrm{Unif}[0,1]
$$

3. 采样中间点：

$$
x \sim p_t(\cdot \mid z)
$$

4. 计算 conditional score：

$$
\nabla_x \log p_t(x \mid z)
$$

5. 训练：

$$
\mathcal{L}(\theta)
=
\left\|
s_t^\theta(x)
-
\nabla_x \log p_t(x \mid z)
\right\|^2
$$

对于 Gaussian path，可以写成：

$$
x_t = \alpha_t z+\beta_t\epsilon
$$

$$
\mathcal{L}_{\mathrm{DSM}}(\theta)
=
\left\|
s_t^\theta(x_t)
+
\frac{\epsilon}{\beta_t}
\right\|^2
$$

采样阶段：

1. 采样噪声：

$$
X_0 \sim p_{\mathrm{init}}
$$

2. 解 SDE：

$$
dX_t =
\left[
u_t^\theta(X_t)
+
\frac{\sigma_t^2}{2}s_t^\theta(X_t)
\right]dt
+
\sigma_t dW_t
$$

3. 返回：

$$
X_1
$$

---

##### 31. Flow Matching 和 Score Matching 的对比

| 方法 | 学什么 | 网络 | 训练 target | 采样方式 |
|---|---|---|---|---|
| Flow Matching | 速度场 | $u_t^\theta(x)$ | $u_t^{\mathrm{target}}(x\mid z)$ | ODE |
| Score Matching | score function | $s_t^\theta(x)$ | $\nabla_x\log p_t(x\mid z)$ | SDE |
| DDM 特殊情况 | score 和 vector field 可互转 | $s_t^\theta$ 或 $u_t^\theta$ | Gaussian path 下的 target | ODE 或 SDE |

---

##### 32. 第三讲的主线总结

第三讲可以按下面这条线理解。

第二讲已经构造出了两个理论目标：

$$
u_t^{\mathrm{target}}(x)
$$

和：

$$
\nabla_x \log p_t(x)
$$

但是这两个 marginal target 都不好直接算。

第三讲的核心 trick 是：

> 不直接训练 marginal target，而是训练 conditional target。

对于 Flow Matching：

$$
u_t^\theta(x)
\approx
u_t^{\mathrm{target}}(x\mid z)
$$

虽然训练时使用 conditional vector field，但最优解是 marginal vector field：

$$
u_t^\theta(x)
\approx
u_t^{\mathrm{target}}(x)
$$

对于 Score Matching：

$$
s_t^\theta(x)
\approx
\nabla_x \log p_t(x\mid z)
$$

虽然训练时使用 conditional score，但最优解是 marginal score：

$$
s_t^\theta(x)
\approx
\nabla_x \log p_t(x)
$$

因此，这一讲真正完成了从理论到算法的转变：

> 第二讲构造 target，第三讲把 target 变成可以训练的 loss。

---

##### 33. 最重要的公式速查

###### Flow Matching Loss

$$
\mathcal{L}_{\mathrm{CFM}}(\theta)
=
\mathbb{E}_{z,t,x}
\left[
\left\|
u_t^\theta(x)
-
u_t^{\mathrm{target}}(x \mid z)
\right\|^2
\right]
$$

其中：

$$
z \sim p_{\mathrm{data}}
$$

$$
t \sim \mathrm{Unif}[0,1]
$$

$$
x \sim p_t(\cdot\mid z)
$$

---

###### Gaussian Path

$$
x_t = \alpha_t z+\beta_t\epsilon
$$

$$
\epsilon \sim \mathcal{N}(0,I_d)
$$

---

###### Gaussian Conditional Vector Field

$$
u_t^{\mathrm{target}}(x \mid z)
=
\left(
\dot{\alpha}_t
-
\frac{\dot{\beta}_t}{\beta_t}\alpha_t
\right)z
+
\frac{\dot{\beta}_t}{\beta_t}x
$$

---
###### Score Matching Loss

$$
\mathcal{L}_{\mathrm{DSM}}(\theta)
=
\mathbb{E}_{z,t,x}
\left[
\left\|
s_t^\theta(x)
-
\nabla_x \log p_t(x\mid z)
\right\|^2
\right]
$$

---

###### Gaussian Conditional Score

$$
\nabla_x \log p_t(x\mid z)
=
-\frac{x-\alpha_tz}{\beta_t^2}
$$

---

###### Gaussian Denoising Score Matching Loss

$$
\mathcal{L}_{\mathrm{DSM}}(\theta)
=
\mathbb{E}_{t,z,\epsilon}
\left[
\left\|
s_t^\theta(\alpha_t z+\beta_t\epsilon)
+
\frac{\epsilon}{\beta_t}
\right\|^2
\right]
$$

---

###### Stochastic Sampling SDE

$$
dX_t =
\left[
u_t^\theta(X_t)
+
\frac{\sigma_t^2}{2}s_t^\theta(X_t)
\right]dt
+
\sigma_t dW_t
$$

---

###### DDM 中 score 到 vector field 的转换

$$
u_t^\theta(x)
=
\left(
\beta_t^2\frac{\dot{\alpha}_t}{\alpha_t}
-
\dot{\beta}_t\beta_t
\right)
s_t^\theta(x)
+
\frac{\dot{\alpha}_t}{\alpha_t}x
$$

---

##### 34. 一句话总结

第三讲的核心是：

> Flow Matching 用 conditional vector field 训练出 marginal vector field；Score Matching 用 conditional score 训练出 marginal score。前者得到 ODE 采样算法，后者得到 SDE 采样算法。在 Gaussian probability path，也就是 DDM 中，score 和 vector field 还可以互相转换，所以不一定需要分别训练两个网络。