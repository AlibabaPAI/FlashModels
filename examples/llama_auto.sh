#!/bin/bash
set -ex
SCRIPTPATH=$(realpath $0 | xargs -n1 dirname)
export PYTHONPATH=$PYTHONPATH:$SCRIPTPATH/../


# Auto Sharding on 4 GPUs with mesh shape of [4,1]
CUDA_VISIBLE_DEVICES=0,1,2,3 \
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
--accelerator acc 2>&1 | tee log/auto_sharding.log