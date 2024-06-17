#!/bin/bash
set -ex

./examples/run.sh --model ./hf_models/config/baichuan-13b-base/ --accelerator acc --gc --mbs 2 --fsdp 4
