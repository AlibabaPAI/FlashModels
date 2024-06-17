#!/bin/bash
set -ex

./examples/run.sh --model ./hf_models/config/qwen-7b/ --accelerator cuda --gc --mbs 4 --fsdp 4
