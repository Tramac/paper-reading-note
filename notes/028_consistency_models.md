这种迭代采样过程使得在计算量和速度之间可以比较灵活的做一个权衡。和一些单步生成模型如 GANs、VAEs、normalizing flows 相比，diffusion models 这种迭代生成过程通常需要 10~2000 倍的计算量，这就导致推理时很慢，对于一些对实时性有要求的应用上是不友好的。

本文的目标是创建一个生成模型，以促进高效的单步生成，而不牺牲迭代采样的重要优势，例如在必要时可以用计算量来换取样本质量，以及执行 zero-shot data editing 任务。

<div align="center">
<img width="500" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/9328912f-57fc-495a-8ee3-be55d298eed1">
</div>

Diffusion Models 可以看作是一个 continuous-time（连续时间）扩散过程，其正向过程就是一个的连续时间的随机过程，可以用一个SDE（stochastic differential equation，随机微分方程）描述，于是成了**朗之万动力学**中的经典问题，在数学上有充分证明的理论工具来支撑。其逆向过程具有**与正向过程SDE相同的联合分布**。

上面提到扩散过程的逆向过程可以用一个 SDE 描述（逆向随机过程），事实上，**存在一个确定性过程（用 ODE 描述）也是它的逆向过程**。这个ODE称为**概率流ODE(probability flow ODE, PF ODE)**，它沿着概率流的轨迹（也就是ODE的解函数）建立了从噪声分布中的点 $x(T)∼pT(x(T))$ 与到数据分布中的点 $x(0) \sim p_{T}(x(T)$ 的映射，也就是说**PF ODE建立了高斯分布样本与数据分布样本的映射。**

本文提出学习一个模型，将任何时间步长的任何点映射到轨迹的起点。该模型的一个显着特性是 self-consistency，即**同一轨迹上的点映射到同一初始点**。所以将这个模型称为 consistency models（一致性模型）。

Consistency Models 可以仅通过一次计算将随机噪声（ODE 轨迹的终点 $x_{T}$）转换生成数据样本（ODE 轨迹的初始点 $x_{0}$，也就是原图像）。并且，通过在多个时间点连接 consistency models 的输出，可以提高样本质量并以更多计算为代价执行零样本数据编辑，类似于 diffusion models 中的迭代采样过程。

本文提供了两种训练 consistency models 的方法：
- Distillation from Diffusion Models
- Training in Isolation

2 Diffusion Models

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

3 Consistency Models

本文提出了一种新型的生成模型 -- consistency models，既支持单步生成，也支持多步迭代生成，可以在计算量和生成质量之间做 trade-offs。其训练方式也有两种：从 pre-trained diffusion models 蒸馏和独立训练。

3.1 Definition 定义

Consistency 在 diffusion 的基础上，定义一个 consistency function（一致性函数）：$f:(x_{t},t) \longmapsto x_{\epsilon}$，该函数具有 self-consistency 特性，即：从 PF ODE 中的任意一个 $(x_t,t)$，它的输出是一致的，用公式表示为：$f(x_t, t)=f(x_{t^{'}},t^{'})$，其中$t,t^{'}\in [\epsilon ,T]$。

上述表示可能不太好理解，我们结合下面图一、图二作一个直观描述：
<div align="center">
<img height="200" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/9328912f-57fc-495a-8ee3-be55d298eed1">
<img height="200" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/da1df3b4-86c4-429d-a2c3-7c6339660d15">
</div>

已知一个前向扩散过程（PF ODE），定义一个一致性函数 $f(x_t,t)$，给定扩散过程中的任意一点$(x_t,t)$（噪声图），$f(x_t,t)$ 都可以直接（单步）将 $x_t$ 去噪至原图 $x_0$。
