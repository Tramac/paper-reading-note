# SDXL Turbo: Adversarial Diffusion Distillation
[paper](https://arxiv.org/pdf/2311.17042) | [code](https://github.com/Stability-AI/generative-models)

## 摘要

本文提出了一种新的蒸馏训练方法 -- Adversarial Diffusion Distillation (ADD) ，训练得到的 SDXL Turbo 模型只需要1-4步就能够生成高质量图。利用已有的 diffusion models 作为 teacher 信号，使用 score distillation 方法，结合 adversarial loss，以确保即使在一两个采样步骤的状态下也能确保证图像的保真度。

分析表明，ADD 的单步生成结果明显优于其它的 few-step 方法，如 GANs、LCM 等，并且仅用 4 步采样就能达到 SDXL 的 SOTA 性能。 同时，ADD 是第一个利用基础模型实现单步实时图像生成的方法。

## 1 引言

Diffusion Models 的一个主要问题是由于其多步迭代过程导致推理很慢。

**SDXL Turbo 模型本质上依旧是 SDXL 模型，其网络架构与 SDXL 一致**，可以理解为一种经过蒸馏训练后的SDXL模型，优化的主要是生成图像时的采样步数。

本文的主要贡献：

- 提出 ADD（Adversarial Diffusion Distillation）方法，可以使已有的 pretrained diffusion models 仅通过 1~4 步采样就可以生成高质量图像；
- 该方法将 adversarial training 和 score distillation 结合在一起；
- ADD 性能优于 LCM、LCM-XL 以及 GANs 等方法，单步采样即可生成复杂的图像生成任务；
- 若使用 4 步采样，ADD-XL 的性能甚至优于其 teacher 模型 SDXL；

## 2 背景

为了解决 Diffusion Models 推理慢的问题，有一些 model distillation 方法被提出，比如 progressive distillation 和 guidance distillation，它们可以将采样步骤减少到 4-8 步，但是性能有明显的下降，而且仍然需要一个迭代训练过程。
> progressive distillation：渐进式蒸馏，目标是将一个步骤很多的教师扩散模型蒸馏为一个步骤较少的学生模型，一般通过反复迭代的方式进行。
> guidance distillation

LCM、LCM-LoRA 可以实现单步采样，并且可以即插即用到已有的 SD 模型中（详见 LCM 的解读）。InstaFlow 提出使用 Rectified Flows 来实现更好的蒸馏过程。

上述这些方法都有共同的缺陷：生成的图像存在模糊和伪影问题。

GANs 虽然可以单步生成，但是其性能明显不如 diffusion-based models。

Score Distillation Sampling 也被称为 Score Jacobian Chaining，是最近被提出的一种蒸馏方法，该方法用于将基础 text-to-image 模型的知识提取到 3D 合成模型中。

## 3 Method

### 3.1 Training Procedure

<div align="center">
<img height="500" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/dd4cffe6-7444-489a-918f-eb7a8199d632">
</div>

网络结构由三部分组成：

1）ADD-student， 用 pretrained diffusion models 做初始化，参数记为 $\theta$；
2）discriminator，判别器，参数记为 $\varphi$；
3）DM teacher，一个 diffusion models 作为 teacher，参数冻住，记为 $\psi$；

训练阶段，给定一张原始图片 $x_{0}$，通过前向扩散过程得到噪声图像 $x_{s} = \alpha_{s}x_{0} + \sigma_{s}\epsilon$，送入 ADD-student 生成样本 $\hat{x}_{\theta } (x_{s}, s)$，然后将 $\hat{x}_{\theta }$ 和 $x_{0}$ 送入判别器以计算 adversarial loss。
> 实际实验中，系数 $\alpha_{s}$ 和 $\sigma_{s}$ 是相同的，因为采样步长 $s$ 是从 ${\tau_{1},...,\tau_{1000}}$ 中均匀采样 N 个样本得到的，其中 N = 4，这也决定了推理时需要的迭代次数。

为了从 DM teacher 中蒸馏知识，将 $\hat{x}_{\theta }$ 送入到 teacher 模型中，去噪得到 $\hat{x}_{\psi } (\hat{x}_{\theta,t},t)$，两者计算蒸馏损失 $\mathcal{L}_{distill}$。

### 3.2 Adversarial Loss

借鉴了 GANs 的思想，设计了Hinge loss（支持向量机SVM中常用的损失函数）作为 SDXL Turbo 模型的 adversarial loss，通过一个 Discriminator 来辨别 student 模型（SDXL 1.0 Base）生成的图像和真实的图像，以确保即使在一个或两个采样步数的低步数状态下也能有高图像保真度，**同时避免了其他蒸馏方法中常见的失真或模糊问题**。

### 3.3 Score Distillation Loss

经典的蒸馏损失函数，让一个强力 Diffusion Model 作为 teacher 模型并冻结参数，让 student 模型（SDXL 1.0 Base）的输出和教师模型的输出尽量一致，具体计算方式使用的是**机器学习中经典的L2损失**。

整个训练过程的损失函数可以表示为：

$$\mathcal{L} = \mathcal{L}_{adv}^{G}(\hat{x}_{\theta}(x_{s},s), \phi)  + \lambda \mathcal{L}_{distill}(\hat{x}_{\theta}(x_{s},s), \psi)$$

其中，$\lambda=2.5$。

## 4 实验

### 4.1 消融实验

<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/556413dd-b4d8-4c56-b973-8f2c03cefe1b">
</div>

### 4.2 性能对比

- 单步结果
<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/0db72e2a-3e87-41f8-9fad-4557ab9d73ac">
</div>

- 4 步结果
<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/edeb6d68-d875-4fc1-9c1a-ee47e4c566b7">
</div>

### 4.3 速度对比

<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/cb4310ef-634d-4654-84b1-8b2799010cfa">
</div>

### 4.4 不同蒸馏方法对比

<div align="center">
<img width="600" alt="image" src="https://github.com/Tramac/paper-reading-note/assets/22740819/980e4013-9350-40fd-8615-cbd4d20d05e3">
</div>

### ref:

- https://zhuanlan.zhihu.com/p/643420260
