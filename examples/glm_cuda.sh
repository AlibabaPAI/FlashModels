#!/bin/bash
set -ex

./examples/run.sh --model ./hf_models/config/chatglm2-6b/ --padding_side "left" --accelerator cuda --gc --mbs 2 --fsdp 4
