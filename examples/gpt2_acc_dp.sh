#!/bin/bash
set -ex

./examples/run.sh --model gpt2-medium --accelerator acc --mbs 2 --fsdp 2 --fp16 --max_seq_length 1024
