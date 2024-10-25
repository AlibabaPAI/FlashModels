#!/bin/bash
set -ex

# FSDP
# ./examples/run.sh --model ./hf_models/config/llama-1b --accelerator acc --gc --mbs 4 --fsdp 4

# FSDP + No GC + No FlashAttention +  Global Batch Size 24
CUDA_VISIBLE_DEVICES=4,5,6,7 ./examples/run.sh --model ./hf_models/config/llama-1b --accelerator acc --mbs 6 --fsdp 4 --no_fa --max_seq_length 2048 --bf16 --log_loss --max_steps 200 --data ./data/alpaca_data.json

# TP + SP + No FlashAttention
CUDA_VISIBLE_DEVICES=4,5,6,7 ./examples/run.sh --model ./hf_models/config/llama-1b --accelerator acc --mbs 24 --tp 4 --sp --no_fa --max_seq_length 2048 --bf16 --log_loss --max_steps 200 --data ./data/alpaca_data.json

# DP + No FlashAttention
CUDA_VISIBLE_DEVICES=4,5,6,7 ./examples/run.sh --model ./hf_models/config/llama-1b --accelerator acc --mbs 6 --dp 4 --no_fa --max_seq_length 2048 --bf16 --log_loss --max_steps 200 --data ./data/alpaca_data.json

# FSDP + No GC + No FlashAttention +  Global Batch Size 24
CUDA_VISIBLE_DEVICES=4,5,6,7 ./examples/run.sh --model ./hf_models/config/llama-1b --accelerator acc --mbs 6 --fsdp 4 --no_fa --max_seq_length 2048 --bf16 --log_loss --max_steps 200

# TP + SP + No FlashAttention
CUDA_VISIBLE_DEVICES=4,5,6,7 ./examples/run.sh --model ./hf_models/config/llama-1b --accelerator acc --mbs 24 --tp 4 --sp --no_fa --max_seq_length 2048 --bf16 --log_loss --max_steps 200

# DP + No FlashAttention
CUDA_VISIBLE_DEVICES=4,5,6,7 ./examples/run.sh --model ./hf_models/config/llama-1b --accelerator acc --mbs 6 --dp 4 --no_fa --max_seq_length 2048 --bf16 --log_loss --max_steps 200


# TP
# ./examples/run.sh --model ./hf_models/config/llama-1b --accelerator acc --gc --mbs 24 --tp 4

# PP
# ./examples/run.sh --model ./hf_models/config/llama-1b --accelerator acc --gc --mbs 24 --pp 4

# PP + TP
# ./examples/run.sh --model ./hf_models/config/llama-1b --accelerator acc --gc --mbs 24 --tp 2 --pp=2
