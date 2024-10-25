#!/bin/bash
set -ex

# FSDP
# note: this need transformers>=4.41.0
./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 4 --max_seq_length 4096 --no_fa --gc

# ====SPMD-FSDP-NoFA====
# bf16 fsdp 4
XLA_IR_DEBUG=1 \
XLA_HLO_DEBUG=1 \
XLA_HLO_DEBUG_VERBOSE_STACK=1 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-spmd-no_flash_attn --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 4 --spmd_fsdp --max_seq_length 2048 --no_fa --log_loss

# fp32 fsdp 4
XLA_IR_DEBUG=1 \
XLA_HLO_DEBUG=1 \
XLA_HLO_DEBUG_VERBOSE_STACK=1 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-spmd-no_flash_attn --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 4 --spmd_fsdp --max_seq_length 2048 --no_fa --log_loss --fp32

# bf16 fsdp 1
XLA_IR_DEBUG=1 \
XLA_HLO_DEBUG=1 \
XLA_HLO_DEBUG_VERBOSE_STACK=1 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-spmd-no_flash_attn --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
CUDA_VISIBLE_DEVICES=4 XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 1 --spmd_fsdp --max_seq_length 2048 --no_fa --log_loss

# fp32 fsdp 1
XLA_IR_DEBUG=1 \
XLA_HLO_DEBUG=1 \
XLA_HLO_DEBUG_VERBOSE_STACK=1 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-spmd-no_flash_attn --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
CUDA_VISIBLE_DEVICES=4 XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 1 --spmd_fsdp --max_seq_length 2048 --no_fa --log_loss --fp32

# ====SPMD-FSDP-FA====
# bf16 fsdp 4
XLA_IR_DEBUG=1 \
XLA_HLO_DEBUG=1 \
XLA_HLO_DEBUG_VERBOSE_STACK=1 PJRT_ALLOCATOR_FRACTION=0.8 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-spmd-flash_attn --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 4 --spmd_fsdp --max_seq_length 2048 --log_loss

# fp32 fsdp 4
XLA_IR_DEBUG=1 \
XLA_HLO_DEBUG=1 \
XLA_HLO_DEBUG_VERBOSE_STACK=1 PJRT_ALLOCATOR_FRACTION=0.8 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-spmd-flash_attn --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 4 --spmd_fsdp --max_seq_length 2048 --log_loss --fp32

# bf16 fsdp 1
XLA_IR_DEBUG=1 \
XLA_HLO_DEBUG=1 \
XLA_HLO_DEBUG_VERBOSE_STACK=1 PJRT_ALLOCATOR_FRACTION=0.8 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-spmd-flash_attn --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
CUDA_VISIBLE_DEVICES=0 XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 1 --spmd_fsdp --max_seq_length 2048 --log_loss

# fp32 fsdp 1
XLA_IR_DEBUG=1 \
XLA_HLO_DEBUG=1 \
XLA_HLO_DEBUG_VERBOSE_STACK=1 PJRT_ALLOCATOR_FRACTION=0.8 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-spmd-flash_attn --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
CUDA_VISIBLE_DEVICES=0 XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 1 --spmd_fsdp --max_seq_length 2048 --log_loss --fp32

# ====Non-SPMD-FSDP====

# bf16 fsdp 4
XLA_IR_DEBUG=1 \
XLA_HLO_DEBUG=1 \
XLA_HLO_DEBUG_VERBOSE_STACK=1 PJRT_ALLOCATOR_FRACTION=0.8 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-nonspmd-flash_attn" \
CUDA_VISIBLE_DEVICES=4,5,6,7 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 4 --max_seq_length 2048 --log_loss

# fp32 fsdp 4
XLA_IR_DEBUG=1 \
XLA_HLO_DEBUG=1 \
XLA_HLO_DEBUG_VERBOSE_STACK=1 PJRT_ALLOCATOR_FRACTION=0.8 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-nonspmd-flash_attn" \
CUDA_VISIBLE_DEVICES=4,5,6,7 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 4 --max_seq_length 2048 --log_loss --fp32


# bf16 fsdp 1
XLA_IR_DEBUG=1 \
XLA_HLO_DEBUG=1 \
XLA_HLO_DEBUG_VERBOSE_STACK=1 PJRT_ALLOCATOR_FRACTION=0.8 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-nonspmd-flash_attn" \
CUDA_VISIBLE_DEVICES=4 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 1 --max_seq_length 2048 --log_loss


# ====Non-SPMD-FSDP-NoFA====
# bf16 fsdp 4
XLA_IR_DEBUG=1 \
XLA_HLO_DEBUG=1 \
XLA_HLO_DEBUG_VERBOSE_STACK=1 PJRT_ALLOCATOR_FRACTION=0.8 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-nonspmd-flash_attn" \
CUDA_VISIBLE_DEVICES=4,5,6,7 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 4 --max_seq_length 2048 --log_loss --no_fa



# bf16 fsdp 1
XLA_IR_DEBUG=1 \
XLA_HLO_DEBUG=1 \
XLA_HLO_DEBUG_VERBOSE_STACK=1 PJRT_ALLOCATOR_FRACTION=0.8 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-nonspmd-flash_attn" \
CUDA_VISIBLE_DEVICES=4 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 1 --max_seq_length 2048 --log_loss --no_fa

# fp32 fsdp 1
XLA_IR_DEBUG=1 \
XLA_HLO_DEBUG=1 \
XLA_HLO_DEBUG_VERBOSE_STACK=1 PJRT_ALLOCATOR_FRACTION=0.8 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-nonspmd-flash_attn" \
CUDA_VISIBLE_DEVICES=4 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 1 --max_seq_length 2048 --log_loss --no_fa --fp32


# Zero3 SPMD based fsdp
PJRT_ALLOCATOR_FRACTION=0.8 \
CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --dp 4 --tp 1 --use_zero3 --max_seq_length 2048 --log_loss



# ==== llama3-1b Performance Comparison====

# nonspmd + fa
XLA_FLAGS="--xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
CUDA_VISIBLE_DEVICES=4,5,6,7 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 4 --max_seq_length 2048 --log_loss


# spmd + fa
XLA_FLAGS="--xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 4 --spmd_fsdp --max_seq_length 2048 --log_loss






# ===== llama3-8b Performance Comparison====

# nonspmd + fa
XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-8b-nonspmd-2408080707" \
./examples/run.sh --model ./hf_models/config/llama-3-8b --accelerator acc --mbs 1 --fsdp 8 --max_seq_length 8192
./examples/run.sh --model ./hf_models/config/llama-3-8b --accelerator acc --mbs 1 --fsdp 8 --max_seq_length 8192 --no_fa



CUDA_VISIBLE_DEVICES=4,5,6,7 ./examples/run.sh --model ./hf_models/config/llama-3-8b --accelerator acc --mbs 1 --fsdp 4 --max_seq_length 2048



./examples/run.sh --model ./hf_models/config/llama-3-8b --accelerator acc --mbs 2 --fsdp 8 --max_seq_length 8192 --gc --gc_cnt 10

# spmd + fa
XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-8b-spmd-2408080602 --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-8b --accelerator acc --mbs 1 --fsdp 8 --spmd_fsdp --max_seq_length 8192 --gc --gc_cnt 2


CUDA_VISIBLE_DEVICES=6,7 LOW_CPU_MEM_USAGE=1 PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-8b-spmd-2408301105 --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-8b --mbs 1 --fsdp 2 --spmd_fsdp --max_seq_length 512 --no_fa

CUDA_VISIBLE_DEVICES=6,7 PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-8b-spmd-2408301105 --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-8b --mbs 1 --fsdp 2 --spmd_fsdp --max_seq_length 512 --no_fa --log_loss


CUDA_VISIBLE_DEVICES=6,7 LOW_CPU_MEM_USAGE=1 PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-8b-spmd-2408301105 --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
./examples/run.sh --model ./hf_models/config/llama-3-8b --mbs 1 --fsdp 2 --max_seq_length 512 --no_fa --log_loss

CUDA_VISIBLE_DEVICES=6,7 PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-8b-spmd-2408301105 --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
./examples/run.sh --model ./hf_models/config/llama-3-8b --mbs 1 --fsdp 2 --max_seq_length 512 --no_fa --log_loss

CUDA_VISIBLE_DEVICES=6,7 LOW_CPU_MEM_USAGE=1 PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-70b-spmd-2409040352 --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-70b --mbs 1 --fsdp 2 --spmd_fsdp --max_seq_length 512 --no_fa


LOW_CPU_MEM_USAGE=1 PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-8.75b-spmd-2409101418 --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-70b --mbs 2 --fsdp 8 --spmd_fsdp --max_seq_length 8192 --gc --gc_cnt 10

LOW_CPU_MEM_USAGE=1 PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-8.75b-nonspmd-2409101513" \
./examples/run.sh --model ./hf_models/config/llama-3-70b --mbs 2 --fsdp 8 --max_seq_length 8192 --gc --gc_cnt 10


CUDA_VISIBLE_DEVICES=4,5,6,7 PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-8b-spmd-2409141551-bf16 --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-8b --mbs 1 --fsdp 4 --spmd_fsdp --max_seq_length 512

CUDA_VISIBLE_DEVICES=4,5,6,7 PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-8b-spmd-2409141551-fp32 --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-8b --mbs 1 --fsdp 4 --spmd_fsdp --max_seq_length 512


# spmd fsdp + Ulysses
CUDA_VISIBLE_DEVICES=4,5,6,7 PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-2409231554" \
XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-1b --mbs 1 --fsdp 4 --sp_num 4 --max_seq_length 2048 --spmd_fsdp

# python fsdp + Ulysses
CUDA_VISIBLE_DEVICES=4,5,6,7 PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b" \
./examples/run.sh --model ./hf_models/config/llama-3-1b --mbs 1 --fsdp 4 --sp_num 2 --max_seq_length 1800

CUDA_VISIBLE_DEVICES=4,5,6,7 PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-2409301444" \
./examples/run.sh --model ./hf_models/config/llama-3-1b --mbs 1 --fsdp 2 --tp --max_seq_length 1800

# fsdp + tp + sp + fa
XLA_IR_DEBUG=1 \
XLA_HLO_DEBUG=1 \
XLA_HLO_DEBUG_VERBOSE_STACK=1 \
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-fsdp-tp-sp" \
XLA_USE_SPMD=1  ./examples/run.sh --model ./hf_models/config/llama-3-1b --mbs 1 --dp 2 --use_zero3 --tp 2 --sp --max_seq_length 7200


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-fsdp-tp-sp" \
XLA_USE_SPMD=1  ./examples/run.sh --model ./hf_models/config/llama-3-1b --mbs 1 --dp 4 --use_zero3 --tp 2 --sp --max_seq_length 1800

# spmd fsdp + ulysses
XLA_IR_DEBUG=1 \
XLA_HLO_DEBUG=1 \
XLA_HLO_DEBUG_VERBOSE_STACK=1 \
CUDA_VISIBLE_DEVICES=4,5,6,7 PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-spmd-sp --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner" \
XLA_USE_SPMD=1  ./examples/run.sh --model ./hf_models/config/llama-3-1b --mbs 1 --fsdp 4 --sp_num 2 --spmd_fsdp --max_seq_length 8192


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-2410101704" \
XLA_USE_SPMD=1  ./examples/run.sh --model ./hf_models/config/llama-3-1b --mbs 1 --fsdp 8 --spmd_fsdp --max_seq_length 1800

CUDA_VISIBLE_DEVICES=4,5,6,7 PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-2409231554" \
XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-1b --mbs 1 --fsdp 4 --max_seq_length 2048 --spmd_fsdp

# python fsdp + ulysses
PJRT_ALLOCATOR_FRACTION=0.92 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-1b-bs1-pythonfsdp --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner --xla_gpu_memory_limit_slop_factor=100 --xla_multiheap_size_constraint_per_heap=4294967296" \
./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 1 --fsdp 4 --sp_num 2 --max_seq_length 8192 --gc --gc_cnt 80
