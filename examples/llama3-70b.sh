# ========= seq_len=8192 mbs=1 python-fsdp=32 =========
LOW_CPU_MEM_USAGE=1 PJRT_ALLOCATOR_FRACTION=0.95 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-70b-bs1-pythonfsdp-2409041559 --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner --xla_gpu_memory_limit_slop_factor=100 --xla_multiheap_size_constraint_per_heap=4294967296" \
./examples/run.sh --model ./hf_models/config/llama-3-70b --accelerator acc --mbs 1 --fsdp 32 --max_seq_length 8192 --gc --gc_cnt 80

# ========= seq_len=8192 mbs=1 spmd-fsdp=32 =========
LOW_CPU_MEM_USAGE=1 PJRT_ALLOCATOR_FRACTION=0.95 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-70b-bs1-spmd-2409041559 --xla_disable_hlo_passes=gpu-convert-async-collectives-to-sync,triton-autotuner --xla_gpu_memory_limit_slop_factor=100 --xla_multiheap_size_constraint_per_heap=4294967296" \
XLA_USE_SPMD=1 ./examples/run.sh --model ./hf_models/config/llama-3-70b --accelerator acc --mbs 1 --fsdp 32 --spmd_fsdp --max_seq_length 8192 --gc --gc_cnt 80

# ========= seq_len=8192 mbs=2 python-fsdp=32 =========
LOW_CPU_MEM_USAGE=1 PJRT_ALLOCATOR_FRACTION=0.90 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3-70b-bs2-pythonfsdp-2409061648 --xla_gpu_memory_limit_slop_factor=100 --xla_multiheap_size_constraint_per_heap=8589934592" \
./examples/run.sh --model ./hf_models/config/llama-3-70b --accelerator acc --mbs 2 --fsdp 32 --max_seq_length 8192 --gc --gc_cnt 80 # OPTIMAL