# Consistency Models

## 题目&作者

Consistency models：一致性模型

作者团队来自OpenAI 宋飏，[个人主页](https://yang-song.net/)。

Consistency Models [poster](https://yang-song.net/assets/pdf/ICML2023/consistency.pdf)。

## 摘要


本文提出了一种新的生成模型 -- Consistency Models，旨在解决 Diffusion Models 在生成时由于迭代采样过程导致速度慢的问题。它可以直接从噪声生成高质量图像，既可以支持快速单步生成，也支持多步采样以提高样本质量，并且支持zero-shot data editing，如 image inpainting、colorization 以及 super-resolution，无需针对这些任务进行专门训练。

Consistency Models 有两种训练方式：一是从 pre-trained Diffusion Models 中蒸馏，二是独立训练。实验证明，在蒸馏训练时，Consistency Models在一步和少步生成中均优于已有的从 Diffusion Models 蒸馏技术。而在单独训练时，Consistency Models 作为一种新型生成模型，在某些基准测试中优于现有的一步、非对抗生成模型。

## 1 引言

Diffusion Models 是一种 score-based 生成模型，它的一个关键特征是其迭代采样过程，即从随机噪声中逐步去噪的过程。这种迭代采样过程可以在 compute 和 quality 之间形成一种 trade-off，一般地，迭代步数越多，生成质量越高。

这种迭代采样过程使得和一些单步生成模型如 GANs、VAEs、normalizing flows 相比，diffusion models 生成过程通常需要 10~2000 倍的计算量，这就导致推理时很慢，对于一些对实时性有要求的应用上是不友好的。

Diffusion Models 可以看作是一个 continuous-time（连续时间）扩散过程，其正向过程就是一个的连续时间的随机过程，可以用一个SDE（stochastic differential equation，随机微分方程）描述，这是一个**朗之万动力学**中的经典问题，在数学上有充分证明的理论工具来支撑。其逆向过程具有**与正向过程SDE相同的联合分布**。

上面提到扩散过程的逆向过程可以用一个 SDE 描述（逆向随机过程），事实上，**存在一个确定性过程（用 ODE 描述）也是它的逆向过程**。这个ODE称为**概率流ODE(probability flow ODE, PF ODE)**，它沿着概率流的轨迹（也就是ODE的解函数）建立了从噪声分布中的点 $x(T)∼pT(x(T))$ 与到数据分布中的点 $x(0) \sim p_{T}(x(T)$ 的映射，也就是说**PF ODE建立了高斯分布样本与数据分布样本的映射。**

本文提出学习一个模型，将任何时间步长的任何点映射到轨迹的起点。该模型的一个显着特性是 self-consistency，即**同一轨迹上的点映射到同一初始点**。所以将这个模型称为 consistency models（一致性模型）。

Consistency Models 可以仅通过一次计算将随机噪声（ODE 轨迹的终点 $x_{T}$）转换生成数据样本（ODE 轨迹的初始点 $x_{0}$，也就是原图像）。并且，通过在多个时间点连接 consistency models 的输出，可以提高样本质量并以更多计算为代价执行零样本数据编辑，类似于 diffusion models 中的迭代采样过程。

本文提供了两种训练 consistency models 的方法：

- Distillation from Diffusion Models
- Training in Isolation

## 2 Diffusion Models

Diffusion Models 在前面的 paper reading 中已经介绍过很多次了，但是常看常新，每篇 paper 的解读视角都不一样，本文也是一样。

Consistency models 很大程度上受到 continuous-time diffusion models 理论的启发。Diffusion models 将原始数据 $x_{0}$，分 T 步逐渐添加高斯噪声，形成随机过程 $x_0, ..., x_T$，当 T 趋向于无穷大时，$x_T$ 趋于标准正态分布。逆向地，生成过程是一个从噪声逐步去噪的过程。

设 $p_{data}(x)$ 代表数据分布，diffusion models 的正向随机过程可以用一个 SDE （随机微分方程）描述：

$$dx_t=\mu (x_t,t)dt + \sigma(t)dw_t$$

其中 $t\in [0,T]$，$\mu$ 和 $\sigma$ 分别为偏移和扩散系数，$w_t$ 符合标准的布朗运动。

上述 SDE 的一个显着特性是存在一个 ODE（常微分方程）形式的解轨迹：

$$dx_t=\left [ \mu (x_t,t)-\frac{1}{2} \sigma (t)^2 \bigtriangledown \log_{}{p_t(x_t)} \right ]dt $$

其中，$\bigtriangledown \log_{}{p_t(x)}$ 是$p_t(x)$ 的得分方程，该函数是 diffusion models 的直接或间接学习目标。因此，diffusion models 被认为是 **score-based** 的生成模型。

或者说，diffusion models 其实是训练了一个网络  $s_\phi (x,t)$ 来拟合 $\bigtriangledown \log_{}{p_t(x)}$，然后 diffusion 的采样过程可以理解为给定 $x_{T} \sim \mathcal{N}(0,I)$，有：

$$\frac{\mathrm{d} x_t}{\mathrm{d} t} = -ts_\phi (x_t,t)$$

给定起始点，给定每个点的梯度后，通过迭代计算得到 $x_{\epsilon}$，其中 $\epsilon$ 是一个小正数，（比如 EDM 中使用的 0.002），用 $x_{\epsilon}$ 当做采样得到的样本（去噪后的图像）。这里引入 $\epsilon$ 是为了避免在 $t=0$ 处易出现的数值不稳定。

## 3 Consistency Models

本文提出了一种新型的生成模型 -- consistency models，既支持单步生成，也支持多步迭代生成，可以在计算量和生成质量之间做 trade-offs。其训练方式也有两种：从 pre-trained diffusion models 蒸馏和独立训练。

### 3.1 Definition 定义

Consistency 在 diffusion 的基础上，定义一个 consistency function（一致性函数）：$f:(x_{t},t) \longmapsto x_{\epsilon}$，该函数具有 self-consistency 特性，即：从 PF ODE 中的任意一个 $(x_t,t)$，它的输出是一致的，用公式表示为：$f(x_t, t)=f(x_{t^{'}},t^{'})$，其中$t,t^{'}\in [\epsilon ,T]$。

上述表示可能不太好理解，我们结合下面图一、图二作一个直观描述：

<div align="center">
<img height="200" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/9328912f-57fc-495a-8ee3-be55d298eed1">
<img height="200" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/da1df3b4-86c4-429d-a2c3-7c6339660d15">
</div>


已知一个前向扩散过程（PF ODE），定义一个一致性函数 $f(x_t,t)$，给定扩散过程中的任意一点$(x_t,t)$（噪声图），$f(x_t,t)$ 都可以直接（单步）将 $x_t$ 去噪至原图 $x_0$。

或者说，Consistency Model 在 Diffusion Model 的基础上，新增了一个约束：**从某个样本到某个噪声的加噪轨迹上的每一个点，都可以经过一个函数 $f$ 映射为这条轨迹的起点。** 显然，同一条轨迹上的点经过 $f$ 映射结果都是同一个点。这也是训练 Consistency Model 时所使用的损失约束。

### 3.2 Parameterization 参数化

上节提到的一致性函数具有以下两个特性：

1. $f(x_{\epsilon},\epsilon)=x_{\epsilon}$，即存在一个  boundary condition（边界条件），即对于轨迹的起点 $x_{\epsilon}$，有 $f(x_{\epsilon},\epsilon)=x_{\epsilon}$；
2. 任意 $t_{1},t_{2}\in [\epsilon ,T]$ 满足 $f(x_{t_{1}},t_{1})=f(x_{t_{2}},t_{2})$；

Consistency Model 的目标是找到一个 $f_{\theta}$，能满足上面这两个条件，来拟合 $f$，通常有两种方式对其参数化：

1. 假设有一个深度神经网络，用 $F_{\theta}$ 表示，可以拟合 $f_{\theta}$，则 consistency models 可以被参数化为：

   $$f_{\theta}(x,t) = \left\{\begin{matrix}
    x & t=\epsilon \\
    F_{\theta}(x,t) & t \in (\epsilon ,T)
   \end{matrix}\right.$$

2. 参考 EDM，使用 skip connections，consistency models 可以设计为：

   $$f_{\theta}(x,t) = c_{skip}(t)x + c_{out}(t)F_{\theta}(x,t)$$​

由于第二种方式在形式上与 diffusion models 很接近，所以本文也是采用此方式。

式中 $c_{skip}(t)$ 和 $c_{out}(t)$​ 被设计为：

$$c_{skip}(t) = \frac{\sigma _{data}^{2} }{(t-\epsilon )^{2}+\sigma _{data}^{2}},c_{out}(t)=\frac{\sigma_{data}(t-\epsilon)}{\sqrt{t^2+\sigma_{data}^{2}} }$$

显然，当 $t=\epsilon$ 时，一定满足边界条件，$c_{skip}(0)=1$，$c_{out}(0)=0$​，即第一点性质。

那么第二个性质怎么去拟合呢？难道是随机从轨迹中采样两个点，送入 $f_{\theta}$ 约束其相同吗？

事实上，当已经有一个训练收敛的 Diffusion Model 时，可以通过最小化下面这个 consistency distillation loss 来去拟合第二个性质：

$$\mathcal{L}_{CD}^{N}(\theta,\theta^{-};\phi ):=\mathbb{E}[\lambda(t_{n})d(f_{\theta}(x_{t_{n+1}},t_{n+1}),f_{\theta ^{-}}(\hat{x}_{t_{n}}^{\phi},t_{n}))]$$

即从样本集中采样一个样本，加噪变为 $x_{t_{n+1}}$，然后利用预训练的 Diffusion 模型去一次噪，预测到另外一个点 $\hat{x}_{t_{n}}^{\phi}$，然后计算这两个点送入 $f_{\theta}$​ 后的结果，用特定损失函数约束其一致。

这里的预测过程形式化地写就是，已知 $s_{\phi}(x,t)$ 和 $\frac{dx_{t}}{dt}=-ts_{\phi}(x_{t},t)$，用 ODE Solver $\Phi$ 去解这个 ODE：

$$\hat{x}_{t_{n}}^{\Phi}=x_{t_{n+1}}+(t_{n}-t_{n+1})\Phi(x_{t_{n+1}},t_{n+1};\Phi)$$

比如常用的 Euler solver 就是 $\Phi(x,t;\phi)=-ts_{\theta}(x,t)$。DDIM、DPM++等都是 ODE Solver 的一种。

Consistency Model 论文附录证明了，当最小化 $\mathcal{L}_{CD}^{N}$ 为 0 时，$f_{\theta}$ 和 $f$ 的误差上确界足够小。所以，通过最小化 $\mathcal{L}_{CD}^{N}$ 就能让 $f_{\theta}$ 和 $f$ 足够接近。

### 3.3 Sampling 采样

文中多次出现 sample 这个描述，该如何理解呢？看下面一段话：

“we can generate samples by sampling from the initial distribution $\hat{x}_{T} \sim \mathcal{N}(0,T^2I)$ and then evaluating the consistency model for $\hat{x}_{\epsilon } = f_{\theta}(\hat{x}_{T},T)$”

从初始的高斯分布中采样得到噪声这个过程称为 sampling，利用 $f_{\theta}$​ 从噪声计算得到图像的过程称为 generate samples。需要结合语义看，本文中的 sample 一般就是指的原图像。

在得到了一个合理的 $f$ 的近似 $f_{\theta}$ 后，就可以调用下面的采样算法从噪声生成样本了：

<div align="center">
<img height="200" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/eca67355-c220-4cf3-8ec7-96c0d7e788f9">
</div>

## 4 Training Consistency Models via Distillation

Consistency Models 的第一种训练方式是从预训练好的 diffusion models $s_\phi (x,t)$ 中蒸馏。

理论部分不展开了（其实是数学弱鸡看不太懂了:)），这里直接贴一个算法步骤：

<div align="center">
<img height="250" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/e2127f03-858d-4aeb-bf77-df2fecbf39f1">
</div>

## 5 Training Consistency Models in Isolation

另一种训练方式是不依赖任何预训练模型，直接单独训练，算法如下：

<div align="center">
<img height="250" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/d70dafa0-d060-443b-9e7d-554d0ccf3f18">
</div>

## 6 实验

<div align="center">
<img height="250" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/0c83e7c7-fdb4-484d-be10-d23b8af1ea0d">
</div>

判定两个点的 $f_{\theta}$ 预测结果是否一致时使用的损失函数可以是 MSE、LPIPS 等等损失，实际上，针对自然图片的损失 LPIPS 在 Consistency Model 的训练过程中取得了最好的结果。

<div align="center">
<img height="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/e10a334a-c777-46dd-b669-fc4345423855">
</div>

## 参考

- https://wrong.wang/blog/20231111-consistency-is-all-you-need
- https://yang-song.net/assets/pdf/ICML2023/consistency.pdf
