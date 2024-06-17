---
language:
- zh
- en
tags:
- qwen
pipeline_tag: text-generation
inference: false
---

# Qwen-7B

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/logo.jpg" width="400"/>
<p>
<br>

<p align="center">
        Qwen-7B <a href="https://modelscope.cn/models/qwen/Qwen-7B/summary">🤖 <a> | <a href="https://huggingface.co/Qwen/Qwen-7B">🤗</a>&nbsp ｜ Qwen-7B-Chat <a href="https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary">🤖 <a> | <a href="https://huggingface.co/Qwen/Qwen-7B-Chat">🤗</a>&nbsp | Qwen-7B-Chat-Int4 <a href="https://huggingface.co/Qwen/Qwen-7B-Chat-Int4">🤗</a>
<br>
<a href="https://github.com/QwenLM/Qwen-7B/blob/main/assets/wechat.png">WeChat</a>&nbsp&nbsp | &nbsp&nbsp<a href="https://discord.gg/z3GAxXZ9Ce">Discord</a>&nbsp&nbsp | &nbsp&nbsp<a href="https://modelscope.cn/studios/qwen/Qwen-7B-Chat-Demo/summary">Demo</a>&nbsp ｜ &nbsp<a href="https://github.com/QwenLM/Qwen-7B/blob/main/tech_memo.md">Report</a>
</p>
<br>

## 介绍 (Introduction)

**通义千问-7B（Qwen-7B）**是阿里云研发的通义千问大模型系列的70亿参数规模的模型。Qwen-7B是基于Transformer的大语言模型, 在超大规模的预训练数据上进行训练得到。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。同时，在Qwen-7B的基础上，我们使用对齐机制打造了基于大语言模型的AI助手Qwen-7B-Chat。本仓库为Qwen-7B的仓库。

通义千问-7B（Qwen-7B）主要有以下特点：

1. **大规模高质量训练语料**：使用超过2.2万亿tokens的数据进行预训练，包含高质量中、英、多语言、代码、数学等数据，涵盖通用及专业领域的训练语料。通过大量对比实验对预训练语料分布进行了优化。
2. **强大的性能**：Qwen-7B在多个中英文下游评测任务上（涵盖常识推理、代码、数学、翻译等），效果显著超越现有的相近规模开源模型，甚至在部分指标上相比更大尺寸模型也有较强竞争力。具体评测结果请详见下文。
3. **覆盖更全面的词表**：相比目前以中英词表为主的开源模型，Qwen-7B使用了约15万大小的词表。该词表对多语言更加友好，方便用户在不扩展词表的情况下对部分语种进行能力增强和扩展。

如果您想了解更多关于通义千问7B开源模型的细节，我们建议您参阅[Github代码库](https://github.com/QwenLM/Qwen-7B)。

**Qwen-7B** is the 7B-parameter version of the large language model series, Qwen (abbr. Tongyi Qianwen), proposed by Aibaba Cloud. Qwen-7B is a Transformer-based large language model, which is pretrained on a large volume of data, including web texts, books, codes, etc. Additionally, based on the pretrained Qwen-7B, we release Qwen-7B-Chat, a large-model-based AI assistant, which is trained with alignment techniques. This repository is the one for Qwen-7B.

The features of Qwen-7B include:

1. **Large-scale high-quality training corpora**: It is pretrained on over 2.2 trillion tokens, including Chinese, English, multilingual texts, code, and mathematics, covering general and professional fields. The distribution of the pre-training corpus has been optimized through a large number of ablation experiments.
2. **Competitive performance**: It significantly surpasses existing open-source models of similar scale on multiple Chinese and English downstream evaluation tasks (including commonsense, reasoning, code, mathematics, etc.), and even surpasses some larger-scale models in several benchmarks. See below for specific evaluation results.
3. **More comprehensive vocabulary coverage**: Compared with other open-source models based on Chinese and English vocabularies, Qwen-7B uses a vocabulary of over 150K tokens. This vocabulary is more friendly to multiple languages, enabling users to directly further enhance the capability for certain languages without expanding the vocabulary.

For more details about the open-source model of Qwen-7B, please refer to the [Github](https://github.com/QwenLM/Qwen-7B) code repository.
<br>

## 要求（Requirements）

* python 3.8及以上版本
* pytorch 1.12及以上版本，推荐2.0及以上版本
* 建议使用CUDA 11.4及以上（GPU用户、flash-attention用户等需考虑此选项）
* python 3.8 and above
* pytorch 1.12 and above, 2.0 and above are recommended
* CUDA 11.4 and above are recommended (this is for GPU users, flash-attention users, etc.)
<br>

## 依赖项 (Dependency)

运行Qwen-7B，请确保满足上述要求，再执行以下pip命令安装依赖库

To run Qwen-7B, please make sure you meet the above requirements, and then execute the following pip commands to install the dependent libraries.

```bash
pip install transformers==4.31.0 accelerate tiktoken einops
```

另外，推荐安装`flash-attention`库，以实现更高的效率和更低的显存占用。

In addition, it is recommended to install the `flash-attention` library for higher efficiency and lower memory usage.

```bash
git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
# 下方安装可选，安装可能比较缓慢。
# Below are optional. Installing them might be slow.
# pip install csrc/layer_norm
# pip install csrc/rotary
```
<br>

## 快速使用（Quickstart）

您可以通过以下代码轻松调用：

You can easily call the model with the following code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="cpu", trust_remote_code=True).eval()
# use auto mode, automatically select precision based on the device.
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)

inputs = tokenizer('蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是', return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
# 蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是亚的斯亚贝巴（Addis Ababa）...
```

关于更多的使用说明，请参考我们的[Github repo](https://github.com/QwenLM/Qwen-7B)获取更多信息。

For more information, please refer to our [Github repo](https://github.com/QwenLM/Qwen-7B) for more information.
<br>

## Tokenizer

> 注：作为术语的“tokenization”在中文中尚无共识的概念对应，本文档采用英文表达以利说明。

基于tiktoken的分词器有别于其他分词器，比如sentencepiece分词器。尤其在微调阶段，需要特别注意特殊token的使用。关于tokenizer的更多信息，以及微调时涉及的相关使用，请参阅[文档](https://github.com/QwenLM/Qwen-7B/blob/main/tokenization_note_zh.md)。

Our tokenizer based on tiktoken is different from other tokenizers, e.g., sentencepiece tokenizer. You need to pay attention to special tokens, especially in finetuning. For more detailed information on the tokenizer and related use in fine-tuning, please refer to the [documentation](https://github.com/QwenLM/Qwen-7B/blob/main/tokenization_note.md).
<br>

## 模型细节 (Model)

Qwen-7B模型规模基本情况如下所示：

The details of the model architecture of Qwen-7B are listed as follows:

| Hyperparameter  |  Value |
|:----------------|:-------|
|    n_layers     |     32 |
|     n_heads     |     32 |
|     d_model     |   4096 |
|   vocab size    | 151851 |
| sequence length |   2048 |

在位置编码、FFN激活函数和normalization的实现方式上，我们也采用了目前最流行的做法，
即RoPE相对位置编码、SwiGLU激活函数、RMSNorm（可选安装flash-attention加速）。

在分词器方面，相比目前主流开源模型以中英词表为主，Qwen-7B使用了超过15万token大小的词表。 该词表在GPT-4使用的BPE词表`cl100k_base`基础上，对中文、多语言进行了优化，在对中、英、代码数据的高效编解码的基础上，对部分多语言更加友好，方便用户在不扩展词表的情况下对部分语种进行能力增强。
词表对数字按单个数字位切分。调用较为高效的[tiktoken分词库](https://github.com/openai/tiktoken)进行分词。

我们从部分语种各随机抽取100万个文档语料，以对比不同模型的编码压缩率（以支持100语种的XLM-R为基准值1，越低越好），具体性能见图。

可以看到Qwen-7B在保持中英代码高效解码的前提下，对部分使用人群较多的语种（泰语th、希伯来语he、阿拉伯语ar、韩语ko、越南语vi、日语ja、土耳其语tr、印尼语id、波兰语pl、俄语ru、荷兰语nl、葡萄牙语pt、意大利语it、德语de、西班牙语es、法语fr等）上也实现了较高的压缩率，使得模型在这些语种上也具备较强的可扩展性和较高的训练和推理效率。

在预训练数据方面，Qwen-7B模型一方面利用了部分开源通用语料，
另一方面也积累了海量全网语料以及高质量文本内容，去重及过滤后的语料超过2.2T tokens。
囊括全网文本、百科、书籍、代码、数学及各个领域垂类。

<p align="center">
    <img src="assets/tokenizer.png" style="width: 1200px"/>
<p>

For position encoding, FFN activation function, and normalization methods, we adopt the prevalent practices, i.e., RoPE relative position encoding, SwiGLU for activation function, and RMSNorm for normalization (optional installation of flash-attention for acceleration).

For tokenization, compared to the current mainstream open-source models based on Chinese and English vocabularies, Qwen-7B uses a vocabulary of over 150K tokens. It first considers efficient encoding of Chinese, English, and code data, and is also more friendly to multilingual languages, enabling users to directly enhance the capability of some languages without expanding the vocabulary. It segments numbers by single digit, and calls the [tiktoken](https://github.com/openai/tiktoken) tokenizer library for efficient tokenization.

We randomly selected 1 million document corpus of each language to test and compare the encoding compression rates of different models (with XLM-R, which supports 100 languages, as the base value 1). The specific performance is shown in the figure above.

As can be seen, while ensuring the efficient decoding of Chinese, English, and code, Qwen-7B also achieves a high compression rate for many other languages (such as th, he, ar, ko, vi, ja, tr, id, pl, ru, nl, pt, it, de, es, fr etc.), equipping the model with strong scalability as well as high training and inference efficiency in these languages.

For pre-training data, on the one hand, Qwen-7B uses part of the open-source generic corpus. On the other hand, it uses a massive amount of accumulated web corpus and high-quality text content. The scale of corpus reaches over 2.2T tokens after deduplication and filtration, encompassing web text, encyclopedias, books, code, mathematics, and various domain.
<br>

## 评测效果（Evaluation）

### 中文评测（Chinese Evaluation）

#### C-Eval

[C-Eval](https://arxiv.org/abs/2305.08322)是评测预训练模型中文常识能力的常用测评框架，覆盖人文、社科、理工、其他专业四个大方向共52个学科。
我们按照标准做法，以开发集样本作为few-shot来源，评价Qwen-7B预训练模型的5-shot验证集与测试集准确率。

[C-Eval](https://arxiv.org/abs/2305.08322) is a common evaluation benchmark for testing the common sense capability of pre-trained models in Chinese. It covers 52 subjects in four major directions: humanities, social sciences, STEM, and other specialties. According to the standard practice, we use the development set samples as the source of few-shot, to evaluate the 5-shot validation set and test set accuracy of the Qwen-7B pre-trained model.

在C-Eval验证集上，Qwen-7B模型和其他模型的准确率对比如下：

The accuracy comparison of Qwen-7B and the other models on the C-Eval validation set is shown as follows:

|      Model      |     Avg. |
|:----------------|:--------:|
|    Alpaca-7B    |     28.9 |
|    Vicuna-7B    |     31.2 |
|   ChatGLM-6B    |     37.1 |
|   Baichuan-7B   |     42.7 |
|   ChatGLM2-6B   |     50.9 |
|   InternLM-7B   |     53.4 |
|     ChatGPT     |     53.5 |
|   Claude-v1.3   |     55.5 |
|   **Qwen-7B**   | **60.8** |

在C-Eval测试集上，Qwen-7B预训练模型与其他模型的效果对比如下表所示：

The performance comparison of Qwen-7B and other models on the C-Eval test set is shown in the following table:

| Model                   |   Avg.   | Avg. (Hard) | STEM   | Social Sciences | Humanities | Others |
|:------------------------|:--------:|:-----------:|:------:|:---------------:|:----------:|:------:|
| ChatGLM-6B              |   38.9   |     29.2    |  33.3  |       48.3      |    41.3    |  38.0  |
| Chinese-Alpaca-Plus-13B |   41.5   |     30.5    |  36.6  |       49.7      |    43.1    |  41.2  |
| Baichuan-7B             |   42.8   |     31.5    |  38.2  |       52.0      |    46.2    |  39.3  |
| WestlakeLM-19B          |   44.6   |     34.9    |  41.6  |       51.0      |    44.3    |  44.5  |
| AndesLM-13B             |   46.0   |     29.7    |  38.1  |       61.0      |    51.0    |  41.9  |
| BatGPT-15B-sirius       |   47.0   |     31.9    |  42.7  |       57.5      |    48.6    |  43.6  |
| ChatGLM2-6B             |   51.7   |     37.1    |  48.6  |       60.5      |    51.3    |  49.8  |
| InternLM-7B             |   52.8   |     37.1    |  48.0  |       67.4      |    55.4    |  45.8  |
| Baichuan-13B            |   53.6   |     36.7    |  47.0  |       66.8      |    57.3    |  49.8  |
| Claude-v1.3             |   54.2   |     39.0    |  51.9  |       61.7      |    52.1    |  53.7  |
| ChatGPT                 |   54.4   |     41.4    |  52.9  |       61.8      |    50.9    |  53.6  |
| **Qwen-7B**             | **59.6** |     41.0    |  52.8  |       74.1      |    63.1    |  55.2  |

可以看到，Qwen-7B在同等规模现有模型中取得了最高的分数，甚至相比更大规模模型也具有较强竞争力。

As can be seen, Qwen-7B achieves the best performance out of all existing models with similar scale and even surpasses larger-scale models.

### 英文评测（English Evaluation）

#### MMLU

[MMLU](https://arxiv.org/abs/2009.03300)是目前评测英文综合能力最权威的基准评测之一，同样覆盖了不同学科领域、不同难度层级的57个子任务。

Qwen-7B在MMLU 5-shot准确率表现如下表：

[MMLU](https://arxiv.org/abs/2009.03300) is currently one of the most recognized benchmarks for evaluating English comprehension abilities, covering 57 subtasks across different academic fields and difficulty levels. The MMLU 5-shot accuracy performance of Qwen-7B is shown in the following table:

|     Model     |   Avg.   | STEM | Social Sciences | Humanities | Others |
|:--------------|:--------:|:----:|:---------------:|:----------:|:------:|
|  LLaMA-7B     |   35.1   | 30.5 |       38.3      |    34.0    |  38.1  |
|  Baichuan-7B  |   42.3   | 35.6 |       48.9      |    38.4    |  48.1  |
|  LLaMA2-7B    |   45.3   | 36.4 |       51.2      |    42.9    |  52.2  |
|  LLaMA-13B    |   46.9   | 35.8 |       53.8      |    45.0    |  53.3  |
|  ChatGLM2-6B  |   47.9   | 41.2 |       54.4      |    43.7    |  54.5  |
|  InternLM-7B  |   51.0   |   -  |         -       |      -     |    -   |
|  Baichuan-13B |   51.6   | 41.6 |       60.9      |    47.4    |  58.5  |
|  LLaMA2-13B   |   54.8   | 44.1 |       62.6      |    52.8    |  61.1  |
|  ChatGLM2-12B |   56.2   | 48.2 |       65.1      |    52.6    |  60.9  |
|  **Qwen-7B**  | **56.7** | 47.6 |       65.9      |    51.5    |  64.7  |

在英文方面，Qwen-7B的效果同样超过了目前国内外其他同类开源预训练模型，同样对比更大规模版本的模型也具有较强竞争力。

In terms of English, Qwen-7B also surpasses other similar open-source pre-trained models, and is competitive when compared to larger versions of other models.

### 代码评测（Coding Evaluation）

我们在[HumanEval](https://github.com/openai/human-eval)（0-shot）上对比预训练模型的代码能力，结果如下：

We compared the code capabilities of pre-trained models on [HumanEval](https://github.com/openai/human-eval), and the results are as follows:

| Model         |  Pass@1  |
|:--------------|:--------:|
| Baichuan-7B   |   9.2    |
| ChatGLM2-6B   |   9.2    |
| InternLM-7B   |   10.4   |
| LLaMA-7B      |   10.5   |
| LLaMA2-7B     |   12.8   |
| Baichuan-13B  |   12.8   |
| LLaMA-13B     |   15.8   |
| MPT-7B        |   18.3   |
| LLaMA2-13B    |   18.3   |
| **Qwen-7B**   | **24.4** |

### 数学评测（Mathematics Evaluation）

数学能力使用常用的[GSM8K](https://github.com/openai/grade-school-math)数据集（8-shot）评价：

We compared the math capabilities of pre-trained models on [GSM8K](https://github.com/openai/grade-school-math) (8-shot), and the results are as follows:

| Model         |   Acc.   |
|:--------------|:--------:|
| MPT-7B        |   6.8    |
| Falcon-7B     |   6.8    |
| Baichuan-7B   |   9.7    |
| LLaMA-7B      |   11.0   |
| LLaMA2-7B     |   14.6   |
| LLaMA-13B     |   17.8   |
| Baichuan-13B  |   26.6   |
| LLaMA2-13B    |   28.7   |
| InternLM-7B   |   31.2   |
| ChatGLM2-6B   |   32.4   |
| ChatGLM2-12B  |   40.9   |
| **Qwen-7B**   | **51.6** |

### 翻译评测（Translation Evaluation）

我们使用[WMT22](https://www.statmt.org/wmt22/translation-task.html)中-英（zh-en）和英-中（en-zh）数据集（5-shot BLEU）评测：

We compared the translation capabilities of pre-trained models on [WMT22](https://www.statmt.org/wmt22/translation-task.html) zh-en and en-zh (5-shot BLEU), and the results are as follows:

|    Model    |     Avg. |    zh-en |    en-zh |
|:------------|:--------:|:--------:|:--------:|
| InternLM-7B |     11.8 |      9.0 |     14.5 |
|  LLaMA-7B   |     12.7 |     16.7 |      8.7 |
|  LLaMA-13B  |     15.8 |     19.5 |     12.0 |
|  LLaMA2-7B  |     19.9 |     21.9 |     17.9 |
|  Bloom-7B   |     20.3 |     19.1 |     21.4 |
| LLaMA2-13B  |     23.3 |     22.4 |     24.2 |
| PolyLM-13B  |     23.6 |     20.2 |     27.0 |
| Baichuan-7B |     24.6 |     22.6 |     26.6 |
| **Qwen-7B** | **27.5** | **24.3** | **30.6** |

### 长序列评测（Long-Context Evaluation）

我们引入NTK插值，LogN注意力缩放，窗口注意力等技巧，将模型的上下文长度扩展到8K以上。在arXiv数据上使用PPL指标测试Qwen-7B在不同长度下的表现，结果如下：

**(若要启用NTK和LogN注意力缩放，请将config.json里的`use_dynamic_ntk`和`use_logn_attn`设置为true)**

We introduce NTK-aware interpolation, LogN attention scaling, Window attention, etc. to extend the context length to over 8K tokens. We conduct language modeling experiments on the arXiv dataset with the PPL evaluation. Results are demonstrated below:

**(To use NTK interpolation and LogN scaling, please set `use_dynamic_ntk` and `use_long_attn` to true in config.json.)**

<table>
    <tr>
        <th rowspan="2">Model</th><th colspan="5" align="center">序列长度 Sequence Length</th>
    </tr>
    <tr>
        <th align="center">1024</th><th align="center">2048</th><th align="center">4096</th><th align="center">8192</th><th align="center">16384</th>
    </tr>
    <tr>
        <td>Qwen-7B</td><td align="center"><b>4.23</b></td><td align="center"><b>3.78</b></td><td align="center">39.35</td><td align="center">469.81</td><td align="center">2645.09</td>
    </tr>
    <tr>
        <td>+ dynamic_ntk</td><td align="center"><b>4.23</b></td><td align="center"><b>3.78</b></td><td align="center">3.59</td><td align="center">3.66</td><td align="center">5.71</td>
    </tr>
    <tr>
        <td>+ dynamic_ntk + logn</td><td align="center"><b>4.23</b></td><td align="center"><b>3.78</b></td><td align="center"><b>3.58</b></td><td align="center">3.56</td><td align="center">4.62</td>
    </tr>
    <tr>
        <td>+ dynamic_ntk + logn + window_attn</td><td align="center"><b>4.23</b></td><td align="center"><b>3.78</b></td><td align="center"><b>3.58</b></td><td align="center"><b>3.49</b></td><td align="center"><b>4.32</b></td>
    </tr>
</table>
<br>

## 评测复现（Reproduction）

我们提供了评测脚本，方便大家复现模型效果，详见[链接](https://github.com/QwenLM/Qwen-7B/tree/main/eval)。提示：由于硬件和框架造成的舍入误差，复现结果如有小幅波动属于正常现象。

We have provided evaluation scripts to reproduce the performance of our model, details as [link](https://github.com/QwenLM/Qwen-7B/tree/main/eval).
<br>

## FAQ

如遇到问题，敬请查阅[FAQ](https://github.com/QwenLM/Qwen-7B/blob/main/FAQ_zh.md)以及issue区，如仍无法解决再提交issue。

If you meet problems, please refer to [FAQ](https://github.com/QwenLM/Qwen-7B/blob/main/FAQ.md) and the issues first to search a solution before you launch a new issue.
<br>

## 使用协议（License Agreement）

我们的代码和模型权重对学术研究完全开放，并支持商用。请查看[LICENSE](https://github.com/QwenLM/Qwen-7B/blob/main/LICENSE)了解具体的开源协议细节。如需商用，请填写[问卷](https://dashscope.console.aliyun.com/openModelApply/qianwen)申请。

Our code and checkpoints are open to research purpose, and they are allowed for commercial purposes. Check [LICENSE](https://github.com/QwenLM/Qwen-7B/blob/main/LICENSE) for more details about the license. If you have requirements for commercial use, please fill out the [form](https://dashscope.console.aliyun.com/openModelApply/qianwen) to apply.
<br>

## 联系我们（Contact Us）

如果你想给我们的研发团队和产品团队留言，请通过邮件（qianwen_opensource@alibabacloud.com）联系我们。

If you are interested to leave a message to either our research team or product team, feel free to send an email to qianwen_opensource@alibabacloud.com.

