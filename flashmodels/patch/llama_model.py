import math
import os
from typing import Optional, Tuple

import einops
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
from torch import nn
from torchacc.dist.tp import Mesh, mark_sharding
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (ACT2FN, LlamaRMSNorm,
                                                      LlamaRotaryEmbedding,
                                                      apply_rotary_pos_emb)

import flashmodels.tensor_parallel as tensor_parallel


class Linear3d(nn.Module):
    """ Custom Linear layer"""
    def __init__(self, in_dim, out_dim, keep_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim, keep_dim))

    def forward(self, x):
        return torch.einsum("bij,jkl->bikl", x, self.weight)


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]
        # merged linear layer
        self._acc_mlp = True if os.environ.get(
            "ACC_LLAMA_MLP") == "1" else False
        if self._acc_mlp:
            self.new_up_proj = Linear3d(hidden_size, intermediate_size, 2)

    def forward(self, x):
        if self._acc_mlp:
            x = self.new_up_proj(x)
            x1 = x[..., 0]
            x2 = x[..., 1]
            return self.down_proj(self.act_fn(x1) * x2)

        if tensor_parallel.get_tp_context().enable_sp():
            x = tensor_parallel.fx_mark_sharding(x, ("dp", None, None),
                                                 barrier=True)
            tensor_parallel.fx_register_hook(x, ("dp", None, None))
            tensor_parallel.fx_register_hook(x, ("dp", "tp", None),
                                             barrier=True)

        out = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        if tensor_parallel.get_tp_context().enable_sp():
            # shard on sequence dim.
            tensor_parallel.fx_mark_sharding(out, ("dp", None, None))
            out = tensor_parallel.fx_mark_sharding(out, ("dp", "tp", None),
                                                   barrier=True)
            tensor_parallel.fx_register_hook(out, ("dp", None, None),
                                             barrier=True)

        return out


class CoreAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

    def forward(self, query_states, key_states, value_states, attention_mask):
        bsz = query_states.shape[0]
        q_len = query_states.shape[-2]
        kv_seq_len = key_states.shape[-2]

        attn_weights = torch.einsum("abij,abjk->abik", query_states,
                                    key_states.transpose(2, 3)) / math.sqrt(
                                        self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min,
                             device=attn_weights.device))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(
                                                 query_states.dtype)
        attn_output = torch.einsum("abij,abjk->abik", attn_weights,
                                   value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}")

        return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from "Attention Is All You Need" paper"""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads}).")
        self.q_proj = nn.Linear(self.hidden_size,
                                self.num_heads * self.head_dim,
                                bias=False)
        self.k_proj = nn.Linear(self.hidden_size,
                                self.num_heads * self.head_dim,
                                bias=False)
        self.v_proj = nn.Linear(self.hidden_size,
                                self.num_heads * self.head_dim,
                                bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                self.hidden_size,
                                bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings)
        self.core_attn = CoreAttention(config)

        # mesh for TP
        self.tp_mesh = None
        self.tp_col_mesh = None
        # mesh for Ulysses SP
        self.sp_mesh_3d = None
        self.sp_mesh_4d = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def _create_tp_mesh(self, tp_num, dp_num=1):
        """r create spmd mesh for tp.
        """
        self.dp_num = dp_num
        self.tp_num = tp_num
        devices_ids = np.arange(self.dp_num * self.tp_num)
        self.tp_mesh = Mesh(devices_ids, (self.dp_num, 1, self.tp_num, 1))
        self.tp_col_mesh = Mesh(devices_ids, (self.dp_num, 1, self.tp_num))

    def _create_sp_mesh(self, sp_num):
        """r create spmd mesh for ulysses sp.
        """
        self.sp_num = sp_num
        devices_ids = np.arange(self.sp_num)
        self.sp_mesh_3d = Mesh(devices_ids, (1, self.sp_num, 1))
        self.sp_mesh_4d = Mesh(devices_ids, (1, self.sp_num, 1, 1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:

        if tensor_parallel.get_tp_context().enable_sp():
            hidden_states = tensor_parallel.fx_mark_sharding(
                hidden_states, ("dp", None, None), barrier=True)
            tensor_parallel.fx_register_hook(hidden_states, ("dp", None, None))
            tensor_parallel.fx_register_hook(hidden_states, ("dp", "tp", None),
                                             barrier=True)

        bsz, q_len, _ = hidden_states.size()

        def _grad_shard_tp(grad):
            if self.tp_col_mesh is not None:
                mark_sharding(grad, self.tp_col_mesh, (0, 1, 2))
            return grad

        def _grad_shard_sp_3d(grad):
            mark_sharding(grad, self.sp_mesh_3d, (0, 1, 2))
            return grad

        def _grad_shard_sp_4d(grad):
            mark_sharding(grad, self.sp_mesh_4d, (0, 1, 2, 3))
            xm.optimization_barrier_([grad])
            grad = grad.view(grad.size())
            return grad

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.tp_col_mesh is not None:
            query_states.register_hook(lambda grad: _grad_shard_tp(grad))
            key_states.register_hook(lambda grad: _grad_shard_tp(grad))
            value_states.register_hook(lambda grad: _grad_shard_tp(grad))
        elif tensor_parallel.get_tp_context().is_initialized():
            tensor_parallel.fx_register_hook(query_states, ("dp", None, "tp"))
            tensor_parallel.fx_register_hook(key_states, ("dp", None, "tp"))
            tensor_parallel.fx_register_hook(value_states, ("dp", None, "tp"))

        query_states = query_states.view(bsz, q_len, self.num_heads,
                                         self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_heads,
                                         self.head_dim)

        if self.tp_mesh is not None:
            mark_sharding(query_states, self.tp_mesh, (0, 1, 2, 3))
            mark_sharding(key_states, self.tp_mesh, (0, 1, 2, 3))
            mark_sharding(value_states, self.tp_mesh, (0, 1, 2, 3))
        elif tensor_parallel.get_tp_context().is_initialized():
            tensor_parallel.fx_mark_sharding(query_states,
                                             ("dp", None, "tp", None))
            tensor_parallel.fx_mark_sharding(key_states,
                                             ("dp", None, "tp", None))
            tensor_parallel.fx_mark_sharding(value_states,
                                             ("dp", None, "tp", None))

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.sp_mesh_4d is not None:
            # insert all-to-all
            mark_sharding(query_states, self.sp_mesh_4d, (0, 1, 2, 3))
            mark_sharding(key_states, self.sp_mesh_4d, (0, 1, 2, 3))
            mark_sharding(value_states, self.sp_mesh_4d, (0, 1, 2, 3))
            query_states.register_hook(lambda grad: _grad_shard_sp_4d(grad))
            key_states.register_hook(lambda grad: _grad_shard_sp_4d(grad))
            value_states.register_hook(lambda grad: _grad_shard_sp_4d(grad))

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_output, attn_weights = self.core_attn(query_states, key_states,
                                                   value_states,
                                                   attention_mask)
        if self.sp_mesh_4d is not None:
            mark_sharding(attn_output, self.sp_mesh_4d, (0, 1, 2, 3))
            attn_output.register_hook(lambda grad: _grad_shard_sp_4d(grad))

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        if self.tp_col_mesh is not None:
            mark_sharding(attn_output, self.tp_col_mesh, (0, 1, 2))
            attn_output.register_hook(lambda grad: _grad_shard_tp(grad))
        elif self.sp_mesh_3d is not None:
            # insert all-to-all
            mark_sharding(attn_output, self.sp_mesh_3d, (0, 1, 2))
            attn_output.register_hook(lambda grad: _grad_shard_sp_3d(grad))
        elif tensor_parallel.get_tp_context().is_initialized():
            attn_output = tensor_parallel.fx_mark_sharding(attn_output,
                                                           ("dp", None, "tp"),
                                                           barrier=True)
            tensor_parallel.fx_register_hook(attn_output, ("dp", None, "tp"))

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # shard on sequence dim.
        if tensor_parallel.get_tp_context().enable_sp():
            tensor_parallel.fx_mark_sharding(attn_output, ("dp", None, None))
            attn_output = tensor_parallel.fx_mark_sharding(attn_output,
                                                           ("dp", "tp", None),
                                                           barrier=True)
            tensor_parallel.fx_register_hook(attn_output, ("dp", None, None),
                                             barrier=True)

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size,
                                                     eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # hidden_states = residual + hidden_states
        # Manual type conversion
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        residual = residual.to(torch.float32)
        hidden_states = residual + hidden_states
        hidden_states = hidden_states.to(input_dtype)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # hidden_states = residual + hidden_states
        # Manual type conversion
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        residual = residual.to(torch.float32)
        hidden_states = residual + hidden_states
        hidden_states = hidden_states.to(input_dtype)

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs


def spmd_flash_attn_fwd(
    self,
    hidden_states: torch.Tensor,
    fsdp_num: int,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    from torchacc.ops import spmd_flash_attn_varlen_xla

    bsz, q_len, _ = hidden_states.size()

    query_states = (self.q_proj(hidden_states).view(bsz, q_len, self.num_heads,
                                                    self.head_dim).transpose(
                                                        1, 2))
    key_states = (self.k_proj(hidden_states).view(
        bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2))
    value_states = (self.v_proj(hidden_states).view(
        bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2))

    kv_seq_len = key_states.shape[-2]
    assert past_key_value is None, "past_key_value is not supported"

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)
    assert not output_attentions, "output_attentions is not supported"

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    past_key_value = (key_states, value_states) if use_cache else None

    # See https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py
    # if attention_mask is not None:
    #     value_states = value_states * attention_mask.unsqueeze(1).unsqueeze(-1)
    q = einops.rearrange(query_states, "b h s ... -> (b s) h ...")
    k = einops.rearrange(key_states, "b h s ... -> (b s) h ...")
    v = einops.rearrange(value_states, "b h s ... -> (b s) h ...")
    max_s = q_len
    cu_q_lens = torch.arange(0, (bsz / fsdp_num + 1) * q_len,
                             step=q_len,
                             dtype=torch.int32,
                             device=q.device)
    device_ids = np.array(range(fsdp_num))
    mesh = xs.Mesh(device_ids, (fsdp_num, 1),
                              ('fsdp', 'tensor'))
    partition_spec=('fsdp', None, None)
    output = spmd_flash_attn_varlen_xla(q,
                                   k,
                                   v,
                                   cu_q_lens,
                                   cu_q_lens,
                                   max_s,
                                   max_s,
                                   dropout_p=0.0,
                                   softmax_scale=None,
                                   causal=True,
                                   mesh=mesh,
                                   partition_spec=partition_spec)
    output = einops.rearrange(output, "(b s) ... -> b s ...", b=bsz)

    output = self.o_proj(einops.rearrange(
        output, "b s h d -> b s (h d)"))
    return output, None, past_key_value


def flash_attn_fwd(
    self,
    hidden_states: torch.Tensor,
    fsdp_num: int,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    from torchacc.ops import flash_attn_varlen_xla

    bsz, q_len, _ = hidden_states.size()

    query_states = (self.q_proj(hidden_states).view(bsz, q_len, self.num_heads,
                                                    self.head_dim).transpose(
                                                        1, 2))
    key_states = (self.k_proj(hidden_states).view(
        bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2))
    value_states = (self.v_proj(hidden_states).view(
        bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2))

    kv_seq_len = key_states.shape[-2]
    assert past_key_value is None, "past_key_value is not supported"

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)
    assert not output_attentions, "output_attentions is not supported"

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    past_key_value = (key_states, value_states) if use_cache else None

    # See https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py
    # if attention_mask is not None:
    #     value_states = value_states * attention_mask.unsqueeze(1).unsqueeze(-1)
    q = einops.rearrange(query_states, "b h s ... -> (b s) h ...")
    k = einops.rearrange(key_states, "b h s ... -> (b s) h ...")
    v = einops.rearrange(value_states, "b h s ... -> (b s) h ...")
    max_s = q_len
    cu_q_lens = torch.arange(0, (bsz + 1) * q_len,
                             step=q_len,
                             dtype=torch.int32,
                             device=q.device)
    output = flash_attn_varlen_xla(q,
                                   k,
                                   v,
                                   cu_q_lens,
                                   cu_q_lens,
                                   max_s,
                                   max_s,
                                   dropout_p=0.0,
                                   softmax_scale=None,
                                   causal=True)
    output = einops.rearrange(output, "(b s) ... -> b s ...", b=bsz)

    return self.o_proj(einops.rearrange(
        output, "b s h d -> b s (h d)")), None, past_key_value


def flash_attn_prep_mask(self, attention_mask, input_shape, inputs_embeds,
                         past_key_values_length):
    return attention_mask


def make_causal_mask(input_ids_shape: torch.Size,
                     dtype: torch.dtype,
                     device: torch.device,
                     past_key_values_length: int = 0):
    """Make causal mask used for bi-directional self-attention."""
    bsz, tgt_len = input_ids_shape
    # mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask = torch.full((tgt_len, tgt_len),
                      torch.finfo(dtype).min,
                      device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([
            torch.zeros(
                tgt_len, past_key_values_length, dtype=dtype, device=device),
            mask
        ],
                         dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len,
                                         tgt_len + past_key_values_length)
