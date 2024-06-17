#!/bin/bash
set -ex

bash ./examples/baichuan_cuda.sh
bash ./examples/baichuan_acc.sh

bash ./examples/glm_cuda.sh
bash ./examples/glm_acc.sh

bash ./examples/llama_cuda.sh
bash ./examples/llama_acc.sh

bash ./examples/olmo_cuda.sh
bash ./examples/olmo_acc.sh

bash ./examples/qwen_cuda.sh
bash ./examples/qwen_acc.sh

bash ./examples/gpt2_acc_dp.sh

bash ./examples/gemma_cuda.sh
bash ./examples/gemma_acc.sh
