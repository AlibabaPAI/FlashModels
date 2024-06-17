---
language:
- zh
- en
pipeline_tag: text-generation
inference: false
---
# Baichuan-13B-Base

<!-- Provide a quick summary of what the model is/does. -->

## 介绍
Baichuan-13B-Base为Baichuan-13B系列模型中的预训练版本，经过对齐后的模型可见[Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)。

[Baichuan-13B](https://github.com/baichuan-inc/Baichuan-13B) 是由百川智能继 [Baichuan-7B](https://github.com/baichuan-inc/baichuan-7B) 之后开发的包含 130 亿参数的开源可商用的大规模语言模型，在权威的中文和英文 benchmark 上均取得同尺寸最好的效果。本次发布包含有预训练 ([Baichuan-13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base)) 和对齐 ([Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)) 两个版本。Baichuan-13B 有如下几个特点：

  1. **更大尺寸、更多数据**：Baichuan-13B 在 [Baichuan-7B](https://github.com/baichuan-inc/baichuan-7B) 的基础上进一步扩大参数量到 130 亿，并且在高质量的语料上训练了 1.4 万亿 tokens，超过 LLaMA-13B 40%，是当前开源 13B 尺寸下训练数据量最多的模型。支持中英双语，使用 ALiBi 位置编码，上下文窗口长度为 4096。
  2. **同时开源预训练和对齐模型**：预训练模型是适用开发者的“基座”，而广大普通用户对有对话功能的对齐模型具有更强的需求。因此本次开源我们同时发布了对齐模型（Baichuan-13B-Chat），具有很强的对话能力，开箱即用，几行代码即可简单的部署。
  3. **更高效的推理**：为了支持更广大用户的使用，我们本次同时开源了 int8 和 int4 的量化版本，相对非量化版本在几乎没有效果损失的情况下大大降低了部署的机器资源门槛，可以部署在如 Nvidia 3090 这样的消费级显卡上。
  4. **开源免费可商用**：Baichuan-13B 不仅对学术研究完全开放，开发者也仅需邮件申请并获得官方商用许可后，即可以免费商用。
  5. 
Baichuan-13B-Base is the pre-training version in the Baichuan-13B series of models, and the aligned model can be found at [Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat).

[Baichuan-13B](https://github.com/baichuan-inc/Baichuan-13B) is an open-source, commercially usable large-scale language model developed by Baichuan Intelligence, following [Baichuan-7B](https://github.com/baichuan-inc/baichuan-7B). With 13 billion parameters, it achieves the best performance in standard Chinese and English benchmarks among models of its size. This release includes two versions: pre-training (Baichuan-13B-Base) and alignment (Baichuan-13B-Chat). Baichuan-13B has the following features:

  1. **Larger size, more data**: Baichuan-13B further expands the parameter volume to 13 billion based on [Baichuan-7B](https://github.com/baichuan-inc/baichuan-7B), and has trained 1.4 trillion tokens on high-quality corpora, exceeding LLaMA-13B by 40%. It is currently the model with the most training data in the open-source 13B size. It supports both Chinese and English, uses ALiBi position encoding, and has a context window length of 4096.
  2. **Open-source pre-training and alignment models simultaneously**: The pre-training model is a "base" suitable for developers, while the general public has a stronger demand for alignment models with dialogue capabilities. Therefore, in this open-source release, we also released the alignment model (Baichuan-13B-Chat), which has strong dialogue capabilities and is ready to use. It can be easily deployed with just a few lines of code.
  3. **More efficient inference**: To support a wider range of users, we have open-sourced the INT8 and INT4 quantized versions. The model can be conveniently deployed on consumer GPUs like the Nvidia 3090 with almost no performance loss.
  4. **Open-source, free, and commercially usable**: Baichuan-13B is not only fully open to academic research, but developers can also use it for free commercially after applying for and receiving official commercial permission via email.


## 模型详情

### 模型描述

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** 百川智能(Baichuan Intelligent Technology)
- **Email**: opensource@baichuan-inc.com
- **Language(s) (NLP):** Chinese/English
- **License:** 【Community License for Baichuan-13B Model】([ZH](Baichuan-13B%20模型社区许可协议.pdf)|
[EN](Community%20License%20for%20Baichuan-13B%20Model.pdf))

  **商业用途(For commercial use):** 请通过 [Email](mailto:opensource@baichuan-inc.com) 联系申请书面授权。(Contact us via [Email](mailto:opensource@baichuan-inc.com) above to apply for written authorization.)

### 模型结构

<!-- Provide the basic links for the model. -->

整体模型基于Baichuan-7B，为了获得更好的推理性能，Baichuan-13B 使用了 ALiBi 线性偏置技术，相对于 Rotary Embedding 计算量更小，对推理性能有显著提升；与标准的 LLaMA-13B 相比，生成 2000 个 tokens 的平均推理速度 (tokens/s)，实测提升 31.6%：

| Model       | tokens/s |
|-------------|----------|
| LLaMA-13B   | 19.4     |
| Baichuan-13B| 25.4     |

具体参数和见下表
|     模型名称       | 隐含层维度  | 层数 | 头数 |词表大小 | 总参数量 | 训练数据（tokens） | 位置编码 | 最大长度 |
|-------------------------|-------|------------|------------|-----------------|--------|--------|----------------|---------|
| Baichuan-7B             | 4,096  | 32       | 32   | 64,000    | 7,000,559,616  | 1.2万亿           | [RoPE](https://arxiv.org/abs/2104.09864)    | 4,096    |
| Baichuan-13B             | 5,120 | 40       | 40  | 64,000    | 13,264,901,120   | 1.4万亿           | [ALiBi](https://arxiv.org/abs/2108.12409)    | 4,096

The overall model is based on Baichuan-7B. In order to achieve better inference performance, Baichuan-13B uses ALiBi linear bias technology, which has a smaller computational load compared to Rotary Embedding, and significantly improves inference performance. Compared with the standard LLaMA-13B, the average inference speed (tokens/s) for generating 2000 tokens has been tested to increase by 31.6%:

| Model       | tokens/s |
|-------------|----------|
| LLaMA-13B   | 19.4     |
| Baichuan-13B| 25.4     |

The specific parameters are as follows:
|     Model Name       | Hidden Size  | Num Layers | Num Attention Heads |Vocab Size | Total Params | Training Dats（tokens） | Position Embedding | Max Length |
|-------------------------|-------|------------|------------|-----------------|--------|--------|----------------|---------|
| Baichuan-7B             | 4,096  | 32       | 32   | 64,000    | 7,000,559,616  | 1.2万亿           | [RoPE](https://arxiv.org/abs/2104.09864)    | 4,096    |
| Baichuan-13B             | 5,120 | 40       | 40  | 64,000    | 13,264,901,120   | 1.4万亿           | [ALiBi](https://arxiv.org/abs/2108.12409)    | 4,096

### 免责声明

我们在此声明，我们的开发团队并未基于 Baichuan-13B 模型开发任何应用，无论是在 iOS、Android、网页或任何其他平台。我们强烈呼吁所有使用者，不要利用 Baichuan-13B 模型进行任何危害国家社会安全或违法的活动。另外，我们也要求使用者不要将 Baichuan-13B 模型用于未经适当安全审查和备案的互联网服务。我们希望所有的使用者都能遵守这个原则，确保科技的发展能在规范和合法的环境下进行。

我们已经尽我们所能，来确保模型训练过程中使用的数据的合规性。然而，尽管我们已经做出了巨大的努力，但由于模型和数据的复杂性，仍有可能存在一些无法预见的问题。因此，如果由于使用 Baichuan-13B 开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。

We hereby declare that our development team has not developed any applications based on the Baichuan-13B model, whether on iOS, Android, the web, or any other platform. We strongly urge all users not to use the Baichuan-13B model for any activities that harm national social security or are illegal. In addition, we also ask users not to use the Baichuan-13B model for internet services that have not undergone appropriate security review and filing. We hope that all users will adhere to this principle to ensure that technological development takes place in a regulated and legal environment.

We have done our utmost to ensure the compliance of the data used in the model training process. However, despite our great efforts, due to the complexity of the model and data, there may still be some unforeseen issues. Therefore, we will not take any responsibility for any issues arising from the use of the Baichuan-13B open-source model, including but not limited to data security issues, public opinion risks, or any risks and problems arising from the model being misled, misused, disseminated, or improperly exploited.

## 训练详情

训练具体设置参见[Baichuan-13B](https://github.com/baichuan-inc/Baichuan-13B)。

For specific training settings, please refer to [Baichuan-13B](https://github.com/baichuan-inc/Baichuan-13B).

## 测评结果

### [C-Eval](https://cevalbenchmark.com/index.html#home)

| Model 5-shot            | STEM  | Social Sciences | Humanities | Others | Average |
|-------------------------|:-----:|:---------------:|:----------:|:------:|:-------:|
| Baichuan-7B             | 38.2  | 52.0            | 46.2       | 39.3   | 42.8    |
| Chinese-Alpaca-Plus-13B | 35.2  | 45.6            | 40.0       | 38.2   | 38.8    |
| Vicuna-13B              | 30.5  | 38.2            | 32.5       | 32.5   | 32.8    |
| Chinese-LLaMA-Plus-13B  | 30.3  | 38.0            | 32.9       | 29.1   | 32.1    |
| Ziya-LLaMA-13B-Pretrain | 27.6  | 34.4            | 32.0       | 28.6   | 30.0    |
| LLaMA-13B               | 27.0  | 33.6            | 27.7       | 27.6   | 28.5    |
| moss-moon-003-base (16B)| 27.0  | 29.1            | 27.2       | 26.9   | 27.4    |
| **Baichuan-13B-Base**   | **45.9** | **63.5** | **57.2**    | **49.3** | **52.4** |
| **Baichuan-13B-Chat**   | **43.7** | **64.6** | **56.2**    | **49.2** | **51.5** |


### [MMLU](https://arxiv.org/abs/2009.03300)

| Model 5-shot            | STEM  | Social Sciences | Humanities | Others | Average |
|-------------------------|:-----:|:---------------:|:----------:|:------:|:-------:|
| Vicuna-13B              | 40.4  | 60.5            | 49.5       | 58.4   | 52.0    | 
| LLaMA-13B               | 36.1  | 53.0            | 44.0       | 52.8   | 46.3    |
| Chinese-Alpaca-Plus-13B | 36.9  | 48.9            | 40.5       | 50.5   | 43.9    |
| Ziya-LLaMA-13B-Pretrain | 35.6  | 47.6            | 40.1       | 49.4   | 42.9    |
| Baichuan-7B             | 35.6  | 48.9            | 38.4       | 48.1   | 42.3    |
| Chinese-LLaMA-Plus-13B  | 33.1  | 42.8            | 37.0       | 44.6   | 39.2    |
| moss-moon-003-base (16B)| 22.4  | 22.8            | 24.2       | 24.4   | 23.6    |
| **Baichuan-13B-Base**   | **41.6** | **60.9** | **47.4**    | **58.5** | **51.6** |
| **Baichuan-13B-Chat**   | **40.9** | **60.9** | **48.8**    | **59.0** | **52.1** |
> 说明：我们采用了 MMLU 官方的[评测方案](https://github.com/hendrycks/test)。

### [CMMLU](https://github.com/haonan-li/CMMLU)

| Model 5-shot            | STEM  | Humanities | Social Sciences | Others | China Specific | Average |
|-------------------------|:-----:|:----------:|:---------------:|:------:|:--------------:|:-------:|
| Baichuan-7B             | 34.4  | 47.5       | 47.6            | 46.6   | 44.3           | 44.0    |
| Vicuna-13B              | 31.8  | 36.2       | 37.6            | 39.5   | 34.3           | 36.3    |
| Chinese-Alpaca-Plus-13B | 29.8  | 33.4       | 33.2            | 37.9   | 32.1           | 33.4    |
| Chinese-LLaMA-Plus-13B  | 28.1  | 33.1       | 35.4            | 35.1   | 33.5           | 33.0    |
| Ziya-LLaMA-13B-Pretrain | 29.0  | 30.7       | 33.8            | 34.4   | 31.9           | 32.1    |
| LLaMA-13B               | 29.2  | 30.8       | 31.6            | 33.0   | 30.5           | 31.2    |
| moss-moon-003-base (16B)| 27.2  | 30.4       | 28.8            | 32.6   | 28.7           | 29.6    |
| **Baichuan-13B-Base**   | **41.7** | **61.1** | **59.8** | **59.0**          | **56.4** | **55.3** |
| **Baichuan-13B-Chat**   | **42.8** | **62.6** | **59.7** | **59.0**          | **56.1** | **55.8** |
> 说明：CMMLU 是一个综合性的中文评估基准，专门用于评估语言模型在中文语境下的知识和推理能力。我们采用了其官方的[评测方案](https://github.com/haonan-li/CMMLU)。

## 微信群组
![WeChat](https://github.com/baichuan-inc/Baichuan-13B/blob/main/media/wechat.jpeg?raw=true)
