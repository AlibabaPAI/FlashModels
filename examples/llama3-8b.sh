#!/bin/bash
# [Note]: Commands in this script should be run under FlashModels Folder
# bash ../scripts/launch-training-torchacc.sh


# ========= seq_len=2048 mbs=1 python-fsdp=8 =========
./examples/run.sh --model ./hf_models/config/llama-3-8b --accelerator acc --mbs 1 --fsdp 8 --max_seq_length 2048

# ========= seq_len=8192 mbs=1 spmd-fsdp=8 =========
XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama-3-8b-bs1-spmd-2409041135 --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-8b --accelerator acc --mbs 1 --fsdp 8 --spmd_fsdp --max_seq_length 8192

# ========= seq_len=8192 mbs=1 spmd-fsdp=8 profile =========
XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama-3-8b-bs2-best --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner --xla_gpu_memory_limit_slop_factor=95" \
XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-8b --accelerator acc --mbs 2 --fsdp 8 --spmd_fsdp --max_seq_length 8192 --gc --gc_cnt 1 --profile

# ========= seq_len=8192 mbs=2 spmd-fsdp=8 =========
XLA_FLAGS="--xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner --xla_gpu_memory_limit_slop_factor=97" \
XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-8b --accelerator acc --mbs 2 --fsdp 8 --spmd_fsdp --max_seq_length 8192 --gc --gc_cnt 1 # OPTIMAL

# ========= seq_len=8192 mbs=2 python-fsdp=8 =========
./examples/run.sh --model ./hf_models/config/llama-3-8b --accelerator acc --mbs 2 --fsdp 8 --max_seq_length 8192 --gc --gc_cnt 9
