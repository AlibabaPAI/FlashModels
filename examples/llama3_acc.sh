#!/bin/bash
set -ex

rm -rf compiled_cache/

# FSDP
# note: this need transformers>=4.41.0
./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 4 --max_seq_length 4096 --no_fa --gc

# SPMD-FSDP
# XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --mbs 2 --fsdp 4 --spmd_fsdp --max_seq_length 4096 --no_fa --gc
