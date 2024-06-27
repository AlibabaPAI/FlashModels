import difflib
import inspect
import os
import re
from typing import Any
import torch
import transformers
from flashmodels.logger import logger
from torchacc import patch_fa

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


def patch_llama(use_flash_attn):
    if use_flash_attn:
        patch_fa()
        from transformers.cache_utils import Cache
        def update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool,
        ):
            return None
        transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask = update_causal_mask

    # (wenting.swt): Delete me when merged in transformers
    if bool(int(os.environ.get("LOW_CPU_MEM_USAGE", "0"))):
        rewrite_load()


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
