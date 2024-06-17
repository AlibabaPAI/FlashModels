# Flash Models

Flash Models is a library containing models accelerated by TorchAcc, a PyTorch training acceleration framework based on [PyTorch/XLA](https://github.com/pytorch/xla).

Currently, it hosts common open-source large language models, with plans to expand to include models from other domains such as vision.


## Setup

Clone the code and install the required dependencies:

```shell
# Start a container using the Docker image with TorchAcc.
sudo docker run  --gpus all --net host --ipc host --shm-size 10G -it --rm --cap-add=SYS_PTRACE registry.cn-hangzhou.aliyuncs.com/pai-dlc/acc:r2.3.0-cuda12.1.0-py3.10-nightly bash

# Clone the code and install the requirements.
git clone https://github.com/AlibabaPAI/FlashModels.git
cd ./FlashModels
pip install -r requirements.txt
```

## Training

Each model supports two types of tasks:
* training with TorchAcc
* training without TorchAcc (Pytorch cuda native mode)

Here is an example of llama training tasks on a single worker with multiple devices (GPU or TPU):

* Training with TorchAcc

```shell
./examples/run.sh \
    --model ./hf_models/config/llama-7b \
    --accelerator acc \
    --gc \
    --mbs 24 \
    --fsdp 8 \
    --bf16
```

* Training without TorchAcc

```shell
./examples/run.sh \
    --model ./hf_models/config/llama-7b \
    --accelerator cuda \
    --gc \
    --mbs 8 \
    --fsdp 8 \
    --bf16
```

## Models

Models available in this repository:


| Model    | FSDP | TP   | PP   | GC   | BF16 | FP16 |
|----------|------|------|------|------|------|------|
| LLaMA-2  | ✓    | ✓    | ✓    | ✓    | ✓    | ✓    |
| QWen     | ✓    | ✗    | ✗    | ✓    | ✓    | ✓    |
| ChatGLM  | ✓    | ✗    | ✗    | ✓    | ✓    | ✓    |
| Olmo     | ✓    | ✗    | ✗    | ✓    | ✓    | ✓    |
| Baichuan | ✓    | ✗    | ✗    | ✓    | ✓    | ✓    |
| ChatGLM  | ✓    | ✗    | ✗    | ✓    | ✓    | ✓    |
| Gpt2     | ✓    | ✗    | ✗    | ✓    | ✓    | ✓    |
| Gemma    | ✓    | ✗    | ✗    | ✓    | ✓    | ✓    |


## Performance

TODO
