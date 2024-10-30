import math
import os
import os.path as osp

import torch
import torchacc as ta
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          get_scheduler)

from flashmodels.datasets import get_dataloader
from flashmodels.logger import logger
from flashmodels.utils import get_last_step_from_ckpt

LOW_CPU_MEM_USAGE = bool(int(os.environ.get("LOW_CPU_MEM_USAGE", "0")))
try:
    from torchdistx import deferred_init
except ImportError:
    LOW_CPU_MEM_USAGE = False


def _count_parameters(model):
    if LOW_CPU_MEM_USAGE:
        logger.warning(
            "When set LOW_CPU_MEM_USAGE, model parameters are not being counted."
        )
        return -1
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Builder(object):
    """build model, tokenizer, loader, optimizer and lr_scheduler"""
    def __init__(self, args):
        self.args = args
        self._init_fn = lambda func, *args, **kwargs: func(*args, **kwargs)
        if LOW_CPU_MEM_USAGE:
            # if self.args.sp_num == 1:
            self._init_fn = lambda func, *args, **kwargs: \
                deferred_init.deferred_init(func, *args, **kwargs)
            # else:
            #     logger.warning(
            #         "LOW_CPU_MEM_USAGE with lazy init does not support for ulysses now."
            #     )

    def build_model_dataloader(self):
        if self.args.resume_from_checkpoint and \
              get_last_step_from_ckpt(self.args.ckpt_dir) > 0:
            # Note: Resume model from checkpoint, and resume tokenizer
            # from pretrained.
            model = self.build_model_from_ckpt()
            tokenizer = self.build_tokenizer(self.args.ckpt_dir)
        else:
            model = self.build_model_from_pretrain()
            tokenizer = self.build_tokenizer(self.args.model_name_or_path)
        loader = self.build_loader(model, tokenizer)

        self.print_model_info(model)

        return model, loader, tokenizer

    def build_model_from_ckpt(self):
        config = AutoConfig.from_pretrained(self.args.model_name_or_path,
                                            trust_remote_code=True)

        model = self._init_fn(
            AutoModelForCausalLM.from_config,
            config,
            #  attn_implementation="flash_attention_2" if self.args.use_flash_attn else "eager",
            trust_remote_code=True)
        return model

    def build_model_from_pretrain(self):
        has_weight = False
        if os.path.exists(self.args.model_name_or_path):
            for file in os.listdir(self.args.model_name_or_path):
                if file.endswith(".bin"):
                    has_weight = True
                    break
        else:
            # from hugging face
            has_weight = True
        if has_weight:
            return self._init_fn(
                AutoModelForCausalLM.from_pretrained,
                self.args.model_name_or_path,
                #  attn_implementation="flash_attention_2" if self.args.use_flash_attn else "eager",
                cache_dir=self.args.cache_dir,
                trust_remote_code=True)
        if self.args.local_rank == 0:
            logger.warning("Model weights are not been set, because" \
                           " there is no .bin file in path %s." %    \
                           self.args.model_name_or_path)
        return self.build_model_from_ckpt()

    def build_tokenizer(self, dir):
        return AutoTokenizer.from_pretrained(
            dir,
            cache_dir=self.args.cache_dir,
            model_max_length=self.args.max_seq_length,
            padding_side=self.args.padding_side,
            use_fast=False,
            trust_remote_code=True)

    def build_loader(self, model, tokenizer):
        return get_dataloader(model, tokenizer, self.args)

    def build_optimizer_scheduler(self, model, loader, args):
        optimizer = self.build_optimizer(model, args)
        lr_scheduler = self.build_lr_scheduler(optimizer, loader, args)

        if self.args.fsdp_num > 1 and not self.args.spmd_fsdp:
            optimizer = self._reset_and_flat_param_for_fsdp(model, optimizer)
        if self.args.resume_from_checkpoint and \
                get_last_step_from_ckpt(self.args.ckpt_dir) > 0:
            optimizer, lr_scheduler = self._load_optimizer_scheduler(
                optimizer, lr_scheduler)
        return optimizer, lr_scheduler

    def build_optimizer(self, model, args):
        optimizer_kwargs = {"lr": args.learning_rate}
        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
            # "foreach": False,
        }
        optimizer_kwargs.update(adam_kwargs)

        # TODO(wenting.swt): It is uncertain whether this piece of code is necessary or not
        # and further confirmation is required. It could have been added to improve the accuracy
        # under different wrapping for models. I would prior to determining whether or not to
        # keep this code.
        def get_parameter_names(model, forbidden_layer_types):
            result = []
            for name, child in model.named_children():
                result += [
                    f"{name}.{n}"
                    for n in get_parameter_names(child, forbidden_layer_types)
                    if not isinstance(child, tuple(forbidden_layer_types))
                ]
            result += list(model._parameters.keys())
            return result

        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
        decay_parameters = [
            name for name in decay_parameters if "bias" not in name
        ]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay":
                0.0,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay":
                0.0,
            },
        ]

        optimizer = self._get_optimizer_cls(args)(optimizer_grouped_parameters,
                                                  **optimizer_kwargs)
        return optimizer

    def build_lr_scheduler(self, optimizer, loader, args):
        num_training_steps = len(loader) * args.num_train_epochs
        if args.max_train_steps > 0:
            num_training_steps = args.max_train_steps
            if args.pp_num > 1:
                num_update_steps_per_epoch = len(loader)
            else:
                num_update_steps_per_epoch = len(
                    loader) // args.gradient_accumulation_steps
            args.num_train_epochs = args.max_train_steps // num_update_steps_per_epoch + int(
                args.max_train_steps % num_update_steps_per_epoch > 0)
        warmup_steps = (args.warmup_steps if args.warmup_steps > 0 else
                        math.ceil(num_training_steps * args.warmup_ratio))
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        return lr_scheduler

    def _reset_and_flat_param_for_fsdp(self, model, optimizer):
        if len(optimizer.param_groups) > 1 and self.args.local_rank == 0:
            logger.warning(
                "FSDP Warning: When using FSDP, several parameter groups will be conflated into "
                "a single one due to nested module wrapping and parameter flattening."
            )
        try:
            ret = optimizer.__class__(model.parameters(), **optimizer.defaults)
        except TypeError:
            if "differentiable" in optimizer.defaults:
                defaults = {
                    k: v
                    for k, v in optimizer.defaults.items()
                    if k != "differentiable"
                }
                ret = optimizer.__class__(model.parameters(), **defaults)
            else:
                raise
        return ret

    def _get_optimizer_cls(self, args):
        return torch.optim.AdamW

    def _load_optimizer_scheduler(self, optimizer, lr_scheduler):
        step = get_last_step_from_ckpt(self.args.ckpt_dir)
        if self.args.accelerator == "acc" and self.args.fsdp_num > 1:

            # Each process loads its own shard of state.
            opt_state = osp.join(
                self.args.ckpt_dir, f"optimizer_rank{ta.local_rank()}"
                f"-of-{ta.dist.world_size()}-step-{step}")
            lr_state = osp.join(
                self.args.ckpt_dir, f"scheduler_rank{ta.local_rank()}"
                f"-of-{ta.dist.world_size()}-step-{step}")
            optimizer_state = torch.load(opt_state)
            lr_scheduler_state = torch.load(lr_state)

            optimizer.load_state_dict(optimizer_state)
            lr_scheduler.load_state_dict(lr_scheduler_state)
        else:
            print(
                f"optimizer and lr scheduler were not loaded from the checkpoint,"
                f"as resuming from checkpoint is currently only supported in acc fsdp mode."
            )
        return optimizer, lr_scheduler

    def print_model_info(self, model):
        if self.args.local_rank == 0:
            logger.info("Model structure: \n %s " % model)
            logger.info("\nModel parameters: %d ." % _count_parameters(model))
