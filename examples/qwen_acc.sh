#!/bin/bash
set -ex

# Qwen FSDP
./examples/run.sh --model ./hf_models/config/qwen-7b/ --accelerator acc --gc --mbs 4 --fsdp 4


# Qwen LORA
# ./examples/run.sh \
#     --model ./hf_models/config/qwen-7b/ \
#     --accelerator acc \
#     --gc \
#     --mbs 4 \
#     --fsdp 4 \
#     --lora \
#     --lora_r 8 \
#     --lora_alpha 32 \
#     --lora_dropout 0.05 \
#     --lora_target_modules 'QKV'
