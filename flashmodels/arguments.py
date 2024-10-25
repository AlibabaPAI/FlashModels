import argparse
import os

import torch

from flashmodels.logger import logger
from flashmodels.patch import patch_gemma, patch_llama, patch_peft


def print_args(args):
    logger.info("FlashModels Arguments: ")
    logger.info(" \n".join(f"    {k} = {v}" for k, v in vars(args).items()))


def parse():
    parser = argparse.ArgumentParser(description="Flash Models Arguments")

    # model args
    parser.add_argument("--model_name_or_path",
                        type=str,
                        default="decapoda-research/llama-7b-hf")
    parser.add_argument("--cache_dir", type=str, default="./models/")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument(
        "--model_type",
        type=str,
        default="",
        choices=["gpt", "llama", "glm", "baichuan", "qwen", "olmo"])

    # dataset args
    parser.add_argument("--dataset_name_or_path",
                        type=str,
                        default="./data/wikitext-2-raw-v1.json")
    parser.add_argument("--dataset_config", type=str, default="")
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--padding_side", type=str, default="right")
    parser.add_argument("--disable_train_sampler",
                        action="store_true",
                        help="Disable Train Sampler")

    # accelerator args
    parser.add_argument("--accelerator",
                        type=str,
                        default="acc",
                        choices=["cuda", "acc", "megatron"],
                        help="accelerator name")
    parser.add_argument("--fsdp_num",
                        type=int,
                        default=1,
                        help="Full sharded data parallel Number")
    parser.add_argument("--spmd_fsdp",
                        action="store_true",
                        default=False,
                        help="Use SPMD based FSDP (FSDPv2)")
    parser.add_argument("--gc",
                        action="store_true",
                        default=False,
                        help="Use gradients checkpoint")
    parser.add_argument(
        "--gc_cnt",
        type=int,
        default=None,
        help="Number of decoder layers for gradient checkpointing")
    parser.add_argument("--tp_num",
                        type=int,
                        default=1,
                        help="Tensor Parallel Number")
    parser.add_argument("--sp",
                        action="store_true",
                        default=False,
                        help="Use Sequence Parallelism.")
    parser.add_argument(
        "--sp_reshard_after_forward",
        action="store_true",
        default=False,
        help="To reduce memory usage, reshard weight after forward in TP-SP, \
        and perform an extra all-gather in the backward pass")
    parser.add_argument("--sp_num",
                        type=int,
                        default=1,
                        help="DeepSpeed Ulysses Sequence \
         Parallel Number. ")
    parser.add_argument("--dp_num",
                        type=int,
                        default=1,
                        help="Data Parallel Number")
    parser.add_argument("--pp_num",
                        type=int,
                        default=1,
                        help="Pipeline Parallel Number")
    parser.add_argument("--fp16",
                        action="store_true",
                        help="Run model in fp16 mode.")
    parser.add_argument("--bf16",
                        action="store_true",
                        help="Run model in bfloat16 mode.")
    parser.add_argument("--force_use_syncfree_adam",
                        action="store_true",
                        help="Force to use \
        syncfree.Adam/AdamW for better tracing peformance.")
    parser.add_argument("--use_zero2",
                        action="store_true",
                        help="Use \
        distributed optimizer(ZeRO2) for SPMD-DP.")
    parser.add_argument("--use_zero3",
                        action="store_true",
                        help="Use \
         ZeRO3 for SPMD-DP.")

    # lora
    parser.add_argument("--lora", action="store_true", help="Use lora")
    parser.add_argument("--lora_r",
                        type=int,
                        default=8,
                        help="lora attention dimension")
    parser.add_argument("--lora_alpha",
                        type=int,
                        default=8,
                        help="lora scaling alpha parameter")
    parser.add_argument("--lora_dropout",
                        type=float,
                        default=0.0,
                        help="The dropout probability \
        for Lora layers")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="QKV",
        choices=["QKV", "ALL"],
        help="The modules to apply Lora to. ALL means all linear layers in \
        decoder layer use lora, QKV means only qkv linears use lora")

    # training args
    parser.add_argument("--global_rank", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint",
                        action="store_true",
                        help="Resume from checkpoint, if true,"
                        " load checkpoint from ckpt_dir")
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--ckpt_freq",
                        type=int,
                        default=100,
                        help="The checkpoint frequency of local steps.")
    parser.add_argument("--profile",
                        action="store_true",
                        help="Open pytorch profiler")
    parser.add_argument("--profile_dir", type=str, default="./profile/")
    parser.add_argument("--profile_stop_step",
                        type=int,
                        default=10,
                        help="Maximum profiling steps")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_step", type=int, default=-1)
    parser.add_argument("--learning_rate",
                        type=float,
                        default=2e-5,
                        help="The initial learning rate for AdamW.")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.03,
                        help="Weight decay for AdamW if we apply some.")
    parser.add_argument("--adam_beta1",
                        type=float,
                        default=0.9,
                        help="Beta1 for AdamW optimizer")
    parser.add_argument("--adam_beta2",
                        type=float,
                        default=0.999,
                        help="Beta2 for AdamW optimizer")
    parser.add_argument("--adam_epsilon",
                        type=float,
                        default=1e-8,
                        help="Epsilon for AdamW optimizer.")
    parser.add_argument("--max_grad_norm",
                        type=float,
                        default=1.0,
                        help="Max gradient norm.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Linear warmup over warmup_ratio fraction of total steps.")
    parser.add_argument("--warmup_steps",
                        type=int,
                        default=0,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--padding_strategy",
                        type=str,
                        default="max_length",
                        help="tokenizer padding strategy",
                        choices=["max_length", "longest"])
    parser.add_argument("--max_train_steps",
                        type=int,
                        default=-1,
                        help="Maximum training steps")
    parser.add_argument("--log_loss",
                        action="store_true",
                        help="Print loss when logging steps")

    args = parser.parse_args()

    if args.lora:
        patch_peft()

    if not args.accelerator == "acc":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    args.global_rank = int(os.getenv("RANK", 0))
    args.local_rank = int(os.getenv("LOCAL_RANK", 0))
    args.world_size = int(os.getenv("WORLD_SIZE", 1))
    if args.global_rank != 0:
        args.profile = False

    # mkdir for ckpt_dir
    if len(args.ckpt_dir) > 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)

    # amp checks.
    args.dtype = torch.float
    if args.fp16:
        assert not args.bf16
        args.dtype = torch.half

    if args.bf16:
        assert not args.fp16
        args.dtype = torch.bfloat16

    # DP/MP checks.
    args.mp_num = args.pp_num * args.tp_num  # model parallel size.
    args.dp_num = max(
        1, args.world_size // (args.mp_num * args.fsdp_num * args.sp_num))

    if not args.model_type:
        if "llama" in args.model_name_or_path.lower():
            args.model_type = "llama"
        elif "gpt" in args.model_name_or_path.lower():
            args.model_type = "gpt"
        elif "glm" in args.model_name_or_path.lower():
            args.model_type = "glm"
        elif "baichuan" in args.model_name_or_path.lower():
            args.model_type = "baichuan"
        elif "qwen" in args.model_name_or_path.lower():
            args.model_type = "qwen"
        elif "olmo" in args.model_name_or_path.lower():
            args.model_type = "olmo"
        elif "gemma" in args.model_name_or_path.lower():
            args.model_type = "gemma"
        else:
            raise NotImplementedError(
                f"Unsupported model: {args.model_name_or_path}")

    if args.model_type == "llama" and args.accelerator == 'acc' and (
            args.fp16 or args.bf16):
<<<<<<< HEAD
        patch_llama(fsdp_num=args.fsdp_num, use_tp=(args.tp_num > 1))
=======
        patch_llama(fsdp_num=args.fsdp_num, ulysses_sp_num=args.sp_num, tp_num=args.tp_num, use_tp=(args.tp_num > 1), spmd_fsdp=args.spmd_fsdp)
>>>>>>> c28a3ce... support fa + tp + sp + fsdp
    if args.model_type == "gemma" and args.accelerator == 'acc':
        patch_gemma()

    if args.local_rank == 0:
        print_args(args)

    return args
