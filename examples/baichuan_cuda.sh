#!/bin/bash
set -ex

./examples/run.sh --model ./hf_models/config/baichuan-13b-base/ --accelerator cuda --gc --mbs 2 --fsdp 4
