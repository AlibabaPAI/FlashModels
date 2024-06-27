#!/bin/bash
set -ex

# FSDP
# note: this need transformers>=4.41.0
./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --gc --mbs 2 --fsdp 8 --max_seq_length 4096 --use_flash_attn
