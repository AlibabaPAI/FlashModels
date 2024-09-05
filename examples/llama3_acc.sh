#!/bin/bash
set -ex

# FSDP
# note: this need transformers>=4.41.0

rm -rf hlo_test

export PJRT_ALLOCATOR_FRACTION=0.95
export XLA_FLAGS='--xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner --xla_dump_hlo_as_text --xla_dump_to=./hlo_test --xla_multiheap_size_constraint_per_heap=4294967296'

CUDA_VISIBLE_DEVICES=6,7 \
./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 2 --max_seq_length 2048 --gc --gc_cnt 10

# SPMD-FSDP
# XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 4 --spmd_fsdp --max_seq_length 4096 --no_fa --gc
