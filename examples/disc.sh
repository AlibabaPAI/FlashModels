set -e
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR="debug"

SCRIPTPATH=$(realpath $0 | xargs -n1 dirname)

DATA="./data/wikitext-2-raw-v1.json"

MODEL_NAME_OR_PATH=./hf_models/config/llama-1b

DEVICE_NUM=1

job_name="llama1b-stablehlo"
export TF_CPP_MIN_LOG_LEVEL=0
export XLA_STABLEHLO_COMPILE=1
export XLA_ALLOCATOR_PREALLOCATE=false
export PJRT_ALLOCATOR_PREALLOCATE=false

rm -rf nsys-llama1b*
# rm -rf dump_dir
export XLA_SYNC_WAIT=1
export XLA_TENSOR_ALLOCATOR_MAXSIZE=1024
#export DISC_ENABLE_COMPUTE_INTENSIVE_FUSE=true
export DISC_DEVICE=CUDA
# export DISC_DEBUG=true
export GPU=True
#export LD_LIBRARY_PATH=/workspace/pytorch/xla/third_party/flash-attention/:/workspace/pytorch/xla/third_party/BladeDISC/build:/workspace/pytorch/xla/third_party/BladeDISC/pytorch_blade/bazel-bin/external/org_disc_compiler/mlir/ral/:$LD_LIBRARY_PATH
export XLA_THREAD_POOL_SIZE=1
export XLA_IO_THREAD_POOL_SIZE=1
# export TF_CPP_VMODULE=disc_compile=10

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DISC_ENABLE_DOT_MERGE=false
export DISC_ENABLE_HORIZONTAL_FUSION=false
export ENBALE_DISC_INPUT_OUTPUT_ALIAS=ON

export DISC_DEBUG_DUMP_DIR=dump_dir/dynamic${DYNAMIC}
export CUDA_VISIBLE_DEVICES=${DYNAMIC}

rm -rf ${DISC_DEBUG_DUMP_DIR}
