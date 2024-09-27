import math
import os
from typing import Optional, Tuple, Union, List

import einops
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch.distributed as dist
import transformers
from packaging import version
from torch import nn
from torchacc.dist.tp import Mesh, mark_sharding
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (ACT2FN, LlamaRMSNorm,
                                                      LlamaRotaryEmbedding,
                                                      apply_rotary_pos_emb, LlamaPreTrainedModel)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast

import torchacc.ops.context_parallel as context_parallel

import flashmodels.tensor_parallel as tensor_parallel

class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if os.getenv('CP_SIZE', None):
            position_ids = context_parallel.slice_forward(
                position_ids, 1 , context_parallel.get_intra_cp_process_group())
            inputs_embeds = context_parallel.slice_forward(
                inputs_embeds, 1 , context_parallel.get_intra_cp_process_group())

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if os.getenv('CP_SIZE', None):
            hidden_states = context_parallel.gather_forward_split_backward(
                hidden_states, 1, context_parallel.get_intra_cp_process_group())


        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


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


def flash_attn_fwd(
    self,
    hidden_states: torch.Tensor,
    fsdp_num: int = None,
    ulysses_sp_num: int = None,
    use_spmd: bool = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:

    if ulysses_sp_num > 1 and not use_spmd:
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads,
                                           self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads,
                                           self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads,
                                           self.head_dim)
        cp_func = context_parallel.ulysses
        cp_group = context_parallel.get_context_parallel_group()
        output = cp_func(q, 
                            k,
                            v,
                            torch.tensor([q_len]),
                            torch.tensor([q_len]),
                            dropout_p = 0.0,
                            softmax_scale=None,
                            causal=True,
                            process_group=cp_group)
        return self.o_proj(einops.rearrange(
            output, "b s h d -> b s (h d)")), None, past_key_value
           
    from torchacc.ops import flash_attn_varlen_xla, spmd_flash_attn_varlen_xla
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

    if version.parse(transformers.__version__) >= version.parse('4.36'):
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)
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

    output = None
    if use_spmd:
        cu_q_lens = torch.arange(0, (bsz / fsdp_num + 1) * q_len,
                                 step=q_len,
                                 dtype=torch.int32,
                                 device=q.device)
        device_ids = np.array(range(fsdp_num))
        mesh = xs.Mesh(device_ids, (fsdp_num, 1), ('fsdp', 'tensor'))
        partition_spec = ('fsdp', None, None)
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
    else:
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
