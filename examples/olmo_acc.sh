#!/bin/bash
set -x

# Check if the ai2-olmo package is installed
package=$(pip list | grep ai2-olmo)
if [ -z "$package" ]; then
    echo "The ai2-olmo package is not installed. Installing it now..."
    pip install boto3 cached-path omegaconf rich
    pip install ai2-olmo --no-deps
fi

./examples/run.sh --model ./hf_models/config/OLMo-7B --accelerator acc --gc --mbs 4 --fsdp 4
