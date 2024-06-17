#!/bin/bash
set -x

transformers_version=$(python -c "import transformers; print(transformers.__version__)")
if [[ ! "$transformers_version" == "4.38.0" ]]; then
  echo "Installing transformers version 4.38.0..."
  # NOTICE: Other models need to be switched back to 4.33.0.
  pip install transformers==4.38.0
fi

./examples/run.sh --model ./hf_models/config/gemma-2b --accelerator cuda --gc --mbs 2 --fsdp 4
