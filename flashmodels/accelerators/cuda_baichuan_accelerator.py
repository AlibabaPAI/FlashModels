import functools
import importlib
import os
from types import MethodType

import torch
import torch.distributed as dist
from accelerate.utils import convert_outputs_to_fp32
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import \
    FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from flashmodels.accelerators.accelerator import (Accelerator,
                                                  AcceleratorFactory)


class CUDABaiChuanAccelerator(Accelerator):

    def accelerate(self, model, loader):
        self.setup()
        torch.cuda.set_device(self.args.local_rank)
        device = torch.device("cuda", self.args.local_rank)
        model = model.to(device)

        if self.args.fsdp_num > 1:
            model = self.fsdp(model)

        return model, loader

    def setup(self):
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=self.args.global_rank,
            world_size=self.args.world_size)
        dist.barrier()

    def get_baichuan_cls(self, class_name="BaichuanLayer"):
        module_path = "transformers_modules/modeling_baichuan"
        module_path = module_path.replace(os.path.sep, ".")
        module = importlib.import_module(module_path)

        return getattr(module, class_name)

    def apply_checkpointing(self, model):
        baichuan_block = self.get_baichuan_cls()

        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        check_fn = lambda submodule: isinstance(submodule, baichuan_block)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=check_fn)

    def fsdp(self, model):
        baichuan_block = self.get_baichuan_cls()

        dtype = torch.float32
        if self.args.fp16:
            dtype = torch.float16
        if self.args.bf16:
            dtype = torch.bfloat16

        # (wenting.swt): When using fsdp, autocast for all intermediate calculations.
        # Only the output is float32. This is to align with Stanford Alpaca"s fsdp implementation
        if self.args.fp16 or self.args.bf16:
            model._original_forward = model.forward
            model_forward_func = model.forward.__func__ if hasattr(
                model.forward, "__func__") else model.forward
            new_forward = torch.cuda.amp.autocast(dtype=dtype)(
                model_forward_func)
            model.forward = MethodType(new_forward, model)
            model.forward = MethodType(
                convert_outputs_to_fp32(model.forward.__func__), model)

        # Use auto_wrap_poliy for nested wrapping instead of only a top-level FSDP.
        auto_wrap_policy = ModuleWrapPolicy({
            baichuan_block,
        })

        mixed_precision_policy = None
        if self.args.fp16 or self.args.bf16:
            mixed_precision_policy = MixedPrecision(
                param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

        # Defalut using FULL_SHARD sharding strategy.
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            mixed_precision=mixed_precision_policy,
            forward_prefetch=False,
            backward_prefetch=None)

        # This must be run after the model has been initialized with FSDP.
        if self.args.gc:
            self.apply_checkpointing(model)

        return model

    def gradient_checkpoint(self, model):
        self.apply_checkpointing(model)
        return model


AcceleratorFactory.regist("cuda-baichuan", CUDABaiChuanAccelerator)
