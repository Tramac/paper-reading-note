# ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision

[paper](https://arxiv.org/pdf/2102.03334.pdf) | [talk](https://www.bilibili.com/video/BV14r4y1j74y/?spm_id_from=333.999.0.0)

## 前言

ViLT提出了一个极其简单的做多模态学习的框架结构，它把模态的特征抽取做到了极小化，把主要的计算量都放在了后面的模态融合部分，大大的提高了模型的推理速度，而且让整个方法的建模变得非常简单，在很大的程度上推动了过去一年多模态学习的进展。
它主要就是移除了多模态学习框架中的目标检测（region feature）部分，极大的简化了整个学习框架。
