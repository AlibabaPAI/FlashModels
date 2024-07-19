# AutoSharding in FlashModels

> [!WARNING]
> This feature is currently experimental and may not be fully supported.

## Introduction to AutoSharding

AutoSharding refers to the capability of automatically distributing the training process of a model across multiple devices or machines. This is particularly useful in scenarios where cluster configurations frequently change or when there is no well-established distribution strategy for the model.

The AutoSharding feature of FlashModels is based on SPMD. It can search for distributed strategies and automatically shard the model. The user only needs to specify the **mesh shape** of the devices or machines, and the rest of the work will be done by the AutoSharding feature.

## Example

To enable the AutoSharding training process for the Llama-1B model on 4 GPUs, use the following code:

```bash
SCRIPTPATH=$(realpath $0 | xargs -n1 dirname)
export PYTHONPATH=$PYTHONPATH:$SCRIPTPATH/../

CUDA_VISIBLE_DEVICES=4,5,6,7 \
XLA_AUTO_SPMD_MESH=4,1 \
XLA_AUTO_SPMD=1 \
XLA_USE_SPMD=1 \
XLA_DISABLE_FUNCTIONALIZATION=1 \
PJRT_DEVICE=CUDA \
torchrun \
--nproc_per_node 4 \
--nnodes 1 \
--node_rank 0 \
--master_addr 127.0.0.1 \
--master_port 9011 \
apps/train.py \
--model_name_or_path ./hf_models/config/llama-1b \
--dataset_name_or_path ./data/alpaca_data.json \
--micro_batch_size 24 \
--num_train_epochs 1 \
--max_seq_length 2048 \
--bf16 \
--max_train_steps 200 \
--log_loss \
--accelerator acc > log/auto_sharding.log 2>&1
```

Compared to the usual training process, you need to set the `XLA_AUTO_SPMD_MESH` environment variable to specify the mesh shape and set `XLA_AUTO_SPMD` to enable the AutoSharding feature.

## Experimental Results

We compared the training speed of the Llama-1B model using 4 NVIDIA A100-SXM4-80GB GPUs and the configurations are listed below:

- Global Batch size: 24
- Sequence length: 2048
- Precision: BF16
- Dataset: Alpaca

Four different configurations were evaluated: tensor parallelism (TP) + sequence parallelism (SP), data parallelism (DP), fully sharded data parallelism (FSDP), and AutoSharding.

<details>
<summary>Commands for each configuration</summary>

```bash
# auto sharding
SCRIPTPATH=$(realpath $0 | xargs -n1 dirname)
export PYTHONPATH=$PYTHONPATH:$SCRIPTPATH/../

ACC_FLASH_ATTN=0  \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
XLA_AUTO_SPMD_MESH=4,1 \
XLA_AUTO_SPMD=1 \
XLA_USE_SPMD=1 \
XLA_DISABLE_FUNCTIONALIZATION=1 \
PJRT_DEVICE=CUDA \
torchrun \
--nproc_per_node 4 \
--nnodes 1 \
--node_rank 0 \
--master_addr 127.0.0.1 \
--master_port 9011 \
apps/train.py \
--model_name_or_path ./hf_models/config/llama-1b \
--dataset_name_or_path ./data/alpaca_data.json \
--micro_batch_size 24 \
--num_train_epochs 1 \
--max_seq_length 2048 \
--bf16 \
--max_train_steps 200 \
--log_loss \
--accelerator acc > log/auto_sharding.log 2>&1

# fsdp
CUDA_VISIBLE_DEVICES=0,1,2,3 ./examples/run.sh --model ./hf_models/config/llama-1b --accelerator acc --mbs 6 --fsdp 4 --no_fa --max_seq_length 2048 --bf16 --log_loss --max_steps 200 --data ./data/alpaca_data.json

# tp + sp
CUDA_VISIBLE_DEVICES=0,1,2,3 ./examples/run.sh --model ./hf_models/config/llama-1b --accelerator acc --mbs 24 --tp 4 --sp --no_fa --max_seq_length 2048 --bf16 --log_loss --max_steps 200 --data ./data/alpaca_data.json

# dp
CUDA_VISIBLE_DEVICES=0,1,2,3 ./examples/run.sh --model ./hf_models/config/llama-1b --accelerator acc --mbs 6 --dp 4 --no_fa --max_seq_length 2048 --bf16 --log_loss --max_steps 200 --data ./data/alpaca_data.json
```

</details>

The results are shown in the following figure:

![AutoSharding](./resources/alpaca-result.png)

AutoSharding explores strategies similar to FSDP, but it does not shard the optimizer states and gradients but only the model weights. The training speed of AutoSharding is slightly slower than DP and FSDP.
