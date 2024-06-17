#!/bin/bash
set -ex

./examples/run.sh --model ./hf_models/config/llama-1b --accelerator cuda --gc --mbs 4 --fsdp 4