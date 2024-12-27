# InternVL2: Better than the Best—Expanding Performance Boundaries of Open-Source Multimodal Models with the Progressive Scaling Strategy

[blog](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/) | [zhihu](https://zhuanlan.zhihu.com/p/706547971) | [huggingface](https://huggingface.co/collections/OpenGVLab/internvl20-667d3961ab5eb12c7ed1463e)

## TL;DR

- 渐进式的训练策略，模型尺度覆盖更广
- 多模态的输入和输出
- 更广泛的训练数据，比如医学数据，视频数据

InternVL2 没有 paper，更像是在 InternVL 1.5 上做了一些小的改进。从标题来看：利用渐进式扩展策略拓展开源多模态模型的性能边界。主要强调了渐进式策略。

<center>
    <img src="https://github.com/user-attachments/assets/9f52c82b-6aa2-4313-b5aa-3054c22af27d">
</center>

InternVL2 系列既有适合嵌入式设备部署的 1B 模型，也有性能优先的 108B 模型，覆盖广泛。凭借更大规模的语言模型，InternVL2-Pro 展现出了出色的多模态理解能力，在各项基准测试中与商业闭源模型的性能相当。

InternVL2 系列主要有以下设计：

- **Progressive with larger language models**：提出了一种渐进式对齐的训练策略，从而形成第一个与大型语言模型原生对齐的视觉基础模型。通过采用模型由小到大、数据由粗粒度到细粒度的渐进式训练策略，我们以相对较低的成本完成了大型模型的训练。这种方法在有限的资源下已经表现出了优异的性能。
- **Multimodal input**：模型支持多种输入模式，包括文本、图像、视频和医疗数据。
- **Multitask output**：得益于 VisionLLMv2 的能力，模型支持各种输出格式，例如图像、边界框和蒙版，表现出广泛的多功能性。通过将 MLLM 与多个下游任务解码器连接起来，InternVL2 可以推广到数百个视觉语言任务，同时实现与专家模型相当的性能。

### Model Card

<center>
    <img src="https://github.com/user-attachments/assets/327db1a5-c5ab-4225-bae3-720a0e0cee1a">
</center>

- 采用了和 InternVL 1.5 中一致的动态分辨率策略
- pre-training 阶段变成了只训练 MLP，个人感觉是因为之前的 vision encoder 训的比较好了，直接拿过来用即可
- fine-tune 阶段依然是训练整个模型

### Performance

和商业模型对比：

<center>
    <img src="https://github.com/user-attachments/assets/203aeece-17ca-4f70-bd73-c5c6617bff38">
</center>
