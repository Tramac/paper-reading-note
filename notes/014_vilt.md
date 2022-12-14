# ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision

[paper](https://arxiv.org/pdf/2102.03334.pdf) | [talk](https://www.bilibili.com/video/BV14r4y1j74y/?spm_id_from=333.999.0.0)

## 前言

ViLT提出了一个极其简单的做多模态学习的框架结构，它把模态的特征抽取做到了极小化，把主要的计算量都放在了后面的模态融合部分，大大的提高了模型的推理速度，而且让整个方法的建模变得非常简单，在很大的程度上推动了过去一年多模态学习的进展。

它主要就是通过**移除多模态学习框架中的目标检测**（region feature）部分，极大的简化了整个学习框架。

<img width="530" alt="image" src="https://user-images.githubusercontent.com/22740819/207489981-1a984494-07ec-40e9-9f34-65be3f904f64.png">

如上图所示，通常的多模态学习框架中，对于text部分都是用一个linear embedding来得到word embedding，但是对视觉部分的处理非常不同，可主要分为三类：

- Region Feature：给定一张图像，通过一个CNN，然后再用一些ROI操作抽取得到区域性特征，相当于做了一个目标检测任务，得到的bbox可以想象成文本侧的token，组成序列；其优点是能和文本侧一样有一个离散型的token（也就是bbox），缺点就是由于使用了目标检测，视觉处理耗时占比很大，如上图中整个pipeline耗时900ms，其中视觉的处理占了885ms。
- Grid Feature：
- Patch Projection：ViLT视觉部分耗时只有0.4ms，但是其训练耗时并没有缩短，需要64张32g的V100，训练3天才能完成训练；而且效果和之前的region feature方法相比也比较一般，其最大优势就是推理耗时的大大缩减。

