# Fuyu-8B

[blog](https://www.adept.ai/blog/fuyu-8b) | [huggingface](https://huggingface.co/adept/fuyu-8b)

## TL; DR

无 vision encoder 和 adapter（encoder-free），纯解码器结构的多模态大模型。

## Model Architecture

<div align=center> 	<img src="https://www.adept.ai/images/blog/fuyu-8b/architecture.png"/> </div>

Adept 是一家做 Copilot 创业的公司，要想高效地帮助用户，必须要准确地理解用户正在干什么，这就需要准确地理解屏幕上的图片、表格、文字等内容。

现有的多模态大模型大多是 Visual Encoder + Adapter + LLM 的结构，这些模型通常只能处理固定分辨率的模型结构。对于分辨率或长宽比不同的图像，只能通过缩放、裁剪或填充来对齐，但这无疑会大大损失掉图片中的原有信息。

另外，这些模型一般都需要经过多个不同的训练阶段，比如 Visual Encoder 通常来自 CLIP，它是以对比学习的方式单独训练的，或者先 pre-training 在 finetune，或者多任务训练、不同分辨率训练等。

Fuyu 是一个纯 decoder-only transformer，**没有专门的 vision encoder 和 Adapter**。图片切分成 patch 之后，直接经过线性映射输入到 transformer 中，移除了 position embeddings，patch 之间的换行用 `\n` 来表示。这种简化的结构使得 Fuyu 支持任意图像分辨率的输入，无需单独的高分辨率和低分辨率训练阶段，大大简化了训练和推理过程。

## Evaluation

<div align=center> 	<img src="https://github.com/user-attachments/assets/fab4e475-39e3-49b5-8f62-0d83363e1621"/> </div>

选择了四个最常用的图像理解数据集进行评估。Fuyu 模型在这些指标上表现良好，不过这些数据集主要关注自然图像，和 Fuyu 实际场景有所不同，并没有进行专门的优化。

## 总结

Fuyu-8B 是站在产品的角度设计模型，主打快速响应和多功能性，简化的架构和训练程序不仅降低了部署和扩展的难度，而且还提高了模型的可解释性和透明度。

图像 patch 并映射之后直接送入 transformer，并没有处理和文本之间的语义 gap，能有比较好的效果和之前的认知有些 diff。

没有公开更多的数据和训练细节。
