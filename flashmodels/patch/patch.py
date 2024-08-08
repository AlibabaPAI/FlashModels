import difflib
import inspect
import os
import re
from typing import Any

import torch
import transformers

from flashmodels.logger import logger
from flashmodels.patch.llama_model import (LlamaAttention, LlamaDecoderLayer,
                                           LlamaMLP, flash_attn_fwd, spmd_flash_attn_fwd,
                                           flash_attn_prep_mask,
                                           make_causal_mask)


def rewrite_load():
    """Rewrite `torch.load` in `from_pretrain` in case to use mmap to reduce the CPU
    memory pressure of loading multiple copies of data under multiple processes"""
    source = inspect.getsource(transformers.modeling_utils)
    modified = re.sub(r"torch\.load\((?![^)]*mmap[^)]*\))([^)]*)\)",
                      r"torch.load(\g<1>, mmap=True)", source)
    modified = re.sub(r"partial\(torch.load,(?![^)]*mmap[^)]*\))([^)]*)\)",
                      r"partial(torch.load,\g<1>, mmap=True)", modified)
    if (int(os.environ.get("LOCAL_RANK", 0)) == 0):
        lines = difflib.ndiff(source.split("\n"), modified.split("\n"))
        diff = "\n".join([
            line for line in lines
            if line.startswith("+") or line.startswith("-")
        ])
        logger.warning(
            f"When set LOW_CPU_MEM_USAGE, all the `torch.load` in transfomers.modeling_utils "
            f"are called with `mmap=True`, diff: \n{diff}")
    exec(modified, transformers.modeling_utils.__dict__)


def patch_llama(fsdp_num, use_tp=False):
    transformers.models.llama.modeling_llama._make_causal_mask = make_causal_mask
    if os.getenv("ACC_FLASH_ATTN", "0") == "1":
        transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = flash_attn_prep_mask
        if os.getenv("XLA_USE_SPMD", "0") == "1":
            transformers.models.llama.modeling_llama.LlamaAttention.forward = spmd_flash_attn_fwd
        else:
            transformers.models.llama.modeling_llama.LlamaAttention.forward = flash_attn_fwd
    elif os.environ.get("ACC_LLAMA_TP") == "1":
        transformers.models.llama.modeling_llama.LlamaMLP = LlamaMLP
    if use_tp:
        # use einsum in linear for SPMD TP/Ulysses.
        transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttention
        transformers.models.llama.modeling_llama.LlamaDecoderLayer = LlamaDecoderLayer

    # (wenting.swt): Delete me when merged in transformers
    if bool(int(os.environ.get("LOW_CPU_MEM_USAGE", "0"))):
        rewrite_load()

    # Set the attention_mask in LlamaAttention to None to match the pattern of FlashAttentionRewriter.
    def wrap_for_flash_attention(func):
        def wrapper(*args, **kwargs):
            kwargs["attention_mask"] = None
            kwargs["fsdp_num"] = fsdp_num
            return func(*args, **kwargs)

        return wrapper

    # always attention_mask=None
    transformers.models.llama.modeling_llama.LlamaAttention.forward = wrap_for_flash_attention(
        transformers.models.llama.modeling_llama.LlamaAttention.forward)


def patch_gemma():
    # Set the attention_mask in GemmaAttention to None to match the pattern of FlashAttentionRewriter.
    def wrap_for_flash_attention(func):
        def wrapper(*args, **kwargs):
            kwargs["attention_mask"] = None
            return func(*args, **kwargs)

        return wrapper

    xla_flags = os.getenv('XLA_FLAGS', '').split(' ')
    pattern = r'--xla_gpu_enable_flash_attention=(\w+)'
    for flag in xla_flags:
        match = re.search(pattern, flag)
        if match:
            value = match.group(1)
            if str(value).lower() == "true":
                transformers.models.gemma.modeling_gemma.GemmaAttention.forward = wrap_for_flash_attention(
                    transformers.models.gemma.modeling_gemma.GemmaAttention.
                    forward)


def patch_lora():
    try:
        import peft
        from peft.tuners import lora
    except ImportError:
        logger.errors("import lora fail, please install peft.")

    def _forward_linear(self, x: torch.Tensor, *args: Any,
                        **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            if version.parse(peft.__version__) > version.parse("0.6.2"):
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self._linear(x)
        elif self.merged:
            if version.parse(peft.__version__) > version.parse("0.6.2"):
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self._linear(x)
        else:
            if version.parse(peft.__version__) > version.parse("0.6.2"):
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self._linear(x)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                result += lora_B(lora_A(dropout(x))) * scaling

            result = result.to(torch_result_dtype)
        return result

    # TODO(baole): delete this patch after
    # https://github.com/huggingface/peft/pull/1010 is merged.
    lora.Linear.forward = _forward_linear
