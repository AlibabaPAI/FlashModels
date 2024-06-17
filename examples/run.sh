#!/bin/bash
set -ex

SCRIPTPATH=$(realpath $0 | xargs -n1 dirname)
export PYTHONPATH=$PYTHONPATH:$SCRIPTPATH/../

MBS=48              # micro batch size
SEQLEN=2048         # max sequence length
NUM_EPOCHS=1        # number epoches
MAX_STEPS=-1        # max steps
GA=1                # gradients accumulation number
LOG_INTERVAL=1      # log interval
GC=0                # gradients checkpoint
FP16=0              # float16
BF16=1              # bfloat16
ACCELERATOR="acc"   # accelerator type
DP_NUM=1            # data parallelism number
PP_NUM=1            # pipeline parallelism number
TP_NUM=1            # tensor parallelism number
FSDP_NUM=1          # fsdp number
DATA=./data/wikitext-2-raw-v1.json               # data name or path
MODEL_NAME_OR_PATH="./hf_models/config/llama-1b" # model name or path


OTHER_ARGS=""

HELP_STR=("Usage: bash examples/run.sh [-h|--help] [--accelerator {acc, cuda}] [--model MODEL_NAME_OR_PATH] \n"
    "\t[--data DATASET_NAME_OR_PATH] [--mbs MICRO_BATCH_SIZE] [--max_seq_length MAX_SEQ_LENGTH] \n"
    "\t[--num_train_epochs NUM_TRAIN_EPOCHS] [--max_steps MAX_TRAIN_STEPS] [--pp PP_NUM] [--tp TP_NUM] [--fsdp FSDP_NUM] \n"
    "\t[--ga GRADIENT_ACCUMULATION_STEPS] [--gc] [--bf16] [--fp16] [--fp32] [--log_interval LOG_INTERVAL] \n"
    "\t[other args for apps/train.py] \n"
    "Examples: \n"
    "\tbash examples/run.sh --accelerator cuda --model ./hf_models/config/llama-7b\n"
    "\tbash examples/run.sh --accelerator acc --model ./hf_models/config/llama-7b\n"
    "\tbash examples/run.sh --model ./hf_models/config/llama-1b --dp 2\n"
    "\tbash examples/run.sh --model ./hf_models/config/llama-7b --pp 2 --fsdp 2\n"
    "\tbash examples/run.sh --model ./hf_models/config/llama-7b --mbs 2 --max_steps 10 --gc 0 --fp32 \n"
    "\tbash examples/run.sh --model ./hf_models/config/llama-7b --tp 4 --fsdp 1 --sp \n"
    "\tbash examples/run.sh --model ./hf_models/config/llama-7b --max_steps 15 --profile \n"
)

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
        echo -e "${HELP_STR[@]}"
        exit 0
        ;;
        --accelerator)
        ACCELERATOR="$2"
        shift
        shift
        ;;
        --model)
        MODEL_NAME_OR_PATH="$2"
        shift
        shift
        ;;
        --data)
        DATA="$2"
        shift
        shift
        ;;
        --mbs)
        MBS="$2"
        shift
        shift
        ;;
        --max_seq_length)
        SEQLEN="$2"
        shift
        shift
        ;;
        --num_train_epochs)
        NUM_EPOCHS="$2"
        shift
        shift
        ;;
        --max_steps)
        MAX_STEPS="$2"
        shift
        shift
        ;;
        --dp)
        DP_NUM="$2"
        shift
        shift
        ;;
        --pp)
        PP_NUM="$2"
        shift
        shift
        ;;
        --tp)
        TP_NUM="$2"
        shift
        shift
        ;;
        --fsdp)
        FSDP_NUM="$2"
        shift
        shift
        ;;
        --ga)
        GA="$2"
        shift
        shift
        ;;
        --gc)
        GC=1
        shift
        ;;
        --bf16)
        BF16=1
        FP16=0
        shift
        ;;
        --fp16)
        FP16=1
        BF16=0
        shift
        ;;
        --fp32)
        FP16=0
        BF16=0
        shift
        ;;
        --log_interval)
        LOG_INTERVAL="$2"
        shift
        shift
        ;;
        -*|--*)
        OTHER_ARGS+=" $1"
        shift
        ;;
        *)
        OTHER_ARGS+=" $1"
        shift
        ;;
    esac
done

OPTION_ARGS=""
[[ "$GC" -eq 1 ]] && OPTION_ARGS+="--gc "
[[ "$BF16" -eq 1 ]] && OPTION_ARGS+="--bf16 "
[[ "$FP16" -eq 1 ]] && OPTION_ARGS+="--fp16 "

if [ "$ACCELERATOR" == "cuda" ]; then
    [ "$PP_NUM" -gt 1 ] && echo "Error: Pipeline Parallelism is not supported for cuda accelerator." && exit 1
    [ "$TP_NUM" -gt 1 ] && echo "Error: Tensor Parallelism is not supported for cuda accelerator." && exit 1
fi

if [ "$TP_NUM" -gt "1" ]; then
    export ACC_LLAMA_TP=1
    export XLA_USE_SPMD=1
fi


if [[ "$ACCELERATOR" == "acc" && ( "$FP16" -eq 1 || "$BF16" -eq 1 ) ]]; then
    export ACC_FLASH_ATTN=1
fi

export XLA_PERSISTENT_CACHE_PATH=./compiled_cache/

MODEL_NAME=$(basename $MODEL_NAME_OR_PATH)
JOB_NAME="${MODEL_NAME}_${ACCELERATOR}_bs${MBS}_seqlen${SEQLEN}_bf16-${BF16}_fp16-${FP16}_pp${PP_NUM}_tp${TP_NUM}_fsdp${FSDP_NUM}"


[ -z "$RANK" ] && RANK=0
[ -z "$WORLD_SIZE" ] && WORLD_SIZE=1
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=127.0.0.1
[ -z "$MASTER_PORT" ] && MASTER_PORT=9010

if [ "$WORLD_SIZE" -eq 1 ]; then
    NPROC_PER_NODE=$((FSDP_NUM * TP_NUM * PP_NUM * DP_NUM))
    [ "$NPROC_PER_NODE" -eq 0 ] && echo "Error: NPROC_PER_NODE is zero." && exit 1
else
    NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
fi

mkdir -p log

torchrun --nproc_per_node $NPROC_PER_NODE \
        --nnodes $WORLD_SIZE \
        --node_rank $RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        apps/train.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --dataset_name_or_path $DATA \
        --micro_batch_size $MBS \
        --num_train_epochs $NUM_EPOCHS \
        --max_seq_length $SEQLEN \
        --accelerator $ACCELERATOR \
        --max_train_steps $MAX_STEPS \
        --pp_num $PP_NUM \
        --tp_num $TP_NUM \
        --fsdp_num $FSDP_NUM \
        --gradient_accumulation_steps $GA \
        $OPTION_ARGS \
        $OTHER_ARGS \
        --log_interval $LOG_INTERVAL 2>&1 | tee ./log/${JOB_NAME}.log ; exit ${PIPESTATUS[0]}
