#!/bin/bash
set -ex

# FSDP
./examples/run.sh --model ./hf_models/config/llama-1b --accelerator acc --gc --mbs 4 --fsdp 4 --use_flash_attn

# TP
# ./examples/run.sh --model ./hf_models/config/llama-1b --accelerator acc --gc --mbs 24 --tp 4

# PP
# ./examples/run.sh --model ./hf_models/config/llama-1b --accelerator acc --gc --mbs 24 --pp 4

# PP + TP
# ./examples/run.sh --model ./hf_models/config/llama-1b --accelerator acc --gc --mbs 24 --tp 2 --pp=2
