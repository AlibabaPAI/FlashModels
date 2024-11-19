import functools
import os
import os.path as osp
from types import MethodType

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torchacc as ta
from torch_xla.distributed.fsdp import consolidate_sharded_model_checkpoints
from torchacc import lazy_device
from torchacc.dist.tp import Mesh, mark_sharding
from torchacc.utils.checkpoint import checkpoint_module, gradient_checkpoint

import flashmodels.tensor_parallel as tensor_parallel
from flashmodels.accelerators.accelerator import (Accelerator,
                                                  AcceleratorFactory)
from flashmodels.logger import logger
from flashmodels.utils import get_last_step_from_ckpt

LOW_CPU_MEM_USAGE = bool(int(os.environ.get("LOW_CPU_MEM_USAGE", "0")))
try:
    from torchdistx import deferred_init, fake
except ImportError:
    LOW_CPU_MEM_USAGE = False


class ACCLLAMAAccelerator(Accelerator):
    def __init__(self, args):
        super().__init__(args)
        devices_ids = np.arange(self.args.world_size)  # 4 (0,1) (2,3)
        # init mesh for SPMD
        # TP
        self.tp_row_mesh = None
        self.tp_col_mesh = None
        self.tp_mesh = None
        if self.args.tp_num > 1 and not self.args.pp_num > 1:
            dp_num = self.args.dp_num
            if self.args.fsdp_num > 1:
                dp_num = self.args.fsdp_num

            new_dids = devices_ids.reshape(
                dp_num, self.args.tp_num).transpose().flatten()
            self.tp_row_mesh = Mesh(new_dids, (self.args.tp_num, dp_num))
            self.tp_col_mesh = Mesh(devices_ids, (dp_num, self.args.tp_num))
            self.tp_mesh = Mesh(devices_ids, (dp_num, self.args.tp_num, 1))
        # Ulysses SP
        self.sp_mesh_3d = None
        if self.args.sp_num > 1 and self.args.spmd_fsdp:  # 2
            self.sp_mesh_3d = Mesh(
                devices_ids, ((int)(self.args.world_size / self.args.sp_num),
                              self.args.sp_num, 1))  # [2,2]
            # self.sp_mesh_3d = Mesh(devices_ids, (1, self.args.sp_num, 1))
            # [4,1] ->

    def accelerate(self, model, loader):
        if self.args.lora:
            from peft import LoraConfig, TaskType, get_peft_model

            target_modules = ["q_proj", "k_proj", "v_proj"]
            if self.args.lora_target_modules == "ALL":
                target_modules.extend(
                    ["o_proj", "gate_proj", "up_proj", "down_proj"])

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
            )
            model = get_peft_model(model, peft_config)
            if self.args.local_rank == 0:
                logger.info("Model after lora: \n %s " % model)

        model, loader = self.accelerate_internal(model, loader)

        return model, loader

    def accelerate_internal(self, model, loader):
        model.model.config.use_cache = False

        if self.args.sp_num > 1 and self.args.spmd_fsdp:
            model = self.ulysses(model)
            # return model, loader

        if self.args.tp_num > 1 and self.args.pp_num == 1:
            model = self.tensor_parallel(model)
            # return model, loader

        # TODO: support this in torchacc
        if self.args.resume_from_checkpoint:
            assert self.args.fsdp_num == self.args.world_size, \
                "Currently, only FSDP supports resume_from_checkpoint."
            model = self.resume_from_checkpoint(model)

        if self.args.tp_num > 1 and self.args.pp_num > 1:
            tensor_parallel.get_tp_context().init_mesh(self.args.pp_num,
                                                       self.args.tp_num,
                                                       self.args.dp_num,
                                                       self.args.sp)

        config = self.get_config(model)
        model = ta.accelerate(model, config=config)

        if self.args.tp_num > 1 and self.args.pp_num > 1:
            self.parallel_3d(model._get_underlay_model())

        return model, loader

    def get_config(self, model):
        def _shard_output_callable(output, mesh):
            if not isinstance(output, tuple) and output[
                    'logits'] is not None and torch_xla._XLAC._get_xla_sharding_spec(
                        output['logits']) == '':
                mark_sharding(output['logits'], mesh, ('fsdp', None, None))


        def get_split_points(llama, num_stages):
            split_points = []
            assert llama.config.num_hidden_layers >= num_stages
            decoders_per_rank = llama.config.num_hidden_layers // num_stages
            for i in range(decoders_per_rank, llama.config.num_hidden_layers,
                           decoders_per_rank):
                split_points.append(f"model.layers.{i}")
            if len(split_points) + 1 > num_stages:
                split_points = split_points[:num_stages - 1]
            return split_points

        config = ta.Config()
        config.compute.fp16 = self.args.fp16
        config.compute.bf16 = self.args.bf16

        config.memory.gc = self.args.gc
        if self.args.gc:
            config.memory.gc_cls = {
                "CoreAttention"
            } if self.args.tp_num > 1 else {"LlamaDecoderLayer"}
            config.memory.gc_cnt = self.args.gc_cnt

        config.dist.dp.size = self.args.dp_num
        config.dist.tp.size = self.args.tp_num

        config.dist.pp.size = self.args.pp_num
        config.dist.pp.num_micro_batches = self.args.gradient_accumulation_steps
        config.dist.pp.input_names = ["input_ids", "attention_mask", "labels"]
        config.dist.pp.split_points = get_split_points(model, self.args.pp_num)
        config.dist.pp.broadcast_loss = False if self.args.tp_num > 1 else True

        config.dist.fsdp.size = self.args.fsdp_num
        config.dist.fsdp.wrap_layer_cls = {"LlamaDecoderLayer"}
        config.dist.fsdp.flatten_parameters = not self.args.lora
        config.dist.fsdp.use_spmd = self.args.spmd_fsdp
        config.dist.fsdp.shard_output_callable = _shard_output_callable

        if self.args.tp_num > 1 and self.args.pp_num > 1:
            config.dist.topology = ["pp", "dp", "tp"]

        return config

    def resume_from_checkpoint(self, model):
        last_step = get_last_step_from_ckpt(self.args.ckpt_dir)
        if xm.is_master_ordinal(local=False):
            try:
                consolidate_sharded_model_checkpoints(
                    ckpt_prefix=osp.join(self.args.ckpt_dir, ""),
                    ckpt_suffix=f"rank-*-of-*-step-{last_step}.pth")
            except:
                print(
                    f"Can not find checkpoint with step {last_step} to resume."
                )
                return model
        xm.rendezvous("ckpt_consolidation")
        ckpt_consolidated = torch.load(osp.join(self.args.ckpt_dir,
                                                "_consolidated.pth"),
                                       mmap=True)
        model.load_state_dict(ckpt_consolidated["model"])
        return model

    def ulysses(self, model):
        """r DeepSeed-Ulysses.
        https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-ulysses
        """
        def _grad_shard_sp(grad):
            mark_sharding(grad, self.sp_mesh_3d, (0, 1, None))
            return grad

        def _forward_linear(m, *args):
            h = args[0]
            out = torch.einsum("bij,jk->bik", h, m.weight.T)
            mark_sharding(out, self.sp_mesh_3d, (0, 1, None))
            if out.requires_grad:
                out.register_hook(lambda grad: _grad_shard_sp(grad))
            return out

        def _forward_sp(m, *args, **kwargs):
            h = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            mark_sharding(h, self.sp_mesh_3d, (0, 1, None))
            if h.requires_grad:
                h.register_hook(lambda grad: _grad_shard_sp(grad))
            if len(args) > 0:
                as_list = list(args)
                as_list[0] = h
                args = tuple(as_list)
            else:
                kwargs["hidden_states"] = h
            output = m._original_forward(*args, **kwargs)
            return output

        device = lazy_device()
        model.lm_head.forward = MethodType(_forward_linear, model.lm_head)
        for decoder_layer in model.model.layers:
            # is_torchdistX_deferred_init = (LOW_CPU_MEM_USAGE and any(
            #     fake.is_fake(param) for param in decoder_layer.parameters()))
            # if is_torchdistX_deferred_init:
            #     print("materialize module")
            #     deferred_init.materialize_module(decoder_layer)
            # decoder_layer.to(device)
            # # print(f"after materialize decoder layer fake: {any(fake.is_fake(param) for param in decoder_layer.parameters())}")
            # for param in decoder_layer.parameters():
            #     print(f"param device: {param.device}")

            if hasattr(decoder_layer.self_attn, "_create_sp_mesh"):
                decoder_layer.self_attn._create_sp_mesh(self.args.sp_num)

            decoder_layer._original_forward = decoder_layer.forward
            decoder_layer.forward = \
                MethodType(_forward_sp, decoder_layer)

            # attn linear
            decoder_layer.self_attn.q_proj.forward = \
                MethodType(_forward_linear, decoder_layer.self_attn.q_proj)
            decoder_layer.self_attn.k_proj.forward = \
                MethodType(_forward_linear, decoder_layer.self_attn.k_proj)
            decoder_layer.self_attn.v_proj.forward = \
                MethodType(_forward_linear, decoder_layer.self_attn.v_proj)
            decoder_layer.self_attn.o_proj.forward = \
                MethodType(_forward_linear, decoder_layer.self_attn.o_proj)
            # mlp linear
            decoder_layer.mlp.gate_proj.forward = \
                MethodType(_forward_linear, decoder_layer.mlp.gate_proj)
            decoder_layer.mlp.up_proj.forward = \
                MethodType(_forward_linear, decoder_layer.mlp.up_proj)
            decoder_layer.mlp.down_proj.forward = \
                MethodType(_forward_linear, decoder_layer.mlp.down_proj)

            # mark_sharding(decoder_layer.self_attn.q_proj.weight,
            #               self.sp_mesh_3d, (0, 1))
            # if self.args.gc:
            #     if gc_cnt > 0:
            #         decoder_layer = checkpoint_module(decoder_layer)
            #         gc_cnt -= 1
        # is_torchdistX_deferred_init = (LOW_CPU_MEM_USAGE and any(
        #     fake.is_fake(param) for param in model.parameters()))
        # if is_torchdistX_deferred_init:
        #     deferred_init.materialize_module(
        #         model,
        #         check_fn=lambda k: not isinstance(k, type(model.model.layers[0]
        #                                                   )))
        # model.to(device)
        return model

    def parallel_3d(self, model):
        context = tensor_parallel.get_tp_context()

        def _forward_dp(m, *args, **kwargs):
            def mark_shard(t):
                if t.shape[0] == 1:
                    return t
                r = (None, ) * (len(t.shape) - 1)
                t = t.view(t.size())
                # Prevent the sharding information of "dp" from being propagating by SPMD to the split,
                # which will introduce a collective permute communication.
                context.tp_mark_sharding(t, (None, ) + r)
                context.tp_mark_sharding(t, ("dp", ) + r)
                return t

            args = ta.utils.utils.apply_to_tensors(mark_shard, args)
            kwargs = ta.utils.utils.apply_to_tensors(mark_shard, kwargs)
            return m._original_forward(*args, **kwargs)

        if self.args.dp_num > 1:
            model._original_forward = model.forward
            model.forward = MethodType(_forward_dp, model)

        def _forward_linear(m, *args, old_specs=None, new_specs=None):
            h = args[0]
            return tensor_parallel.PatchedLinearFor3D.apply(
                h, m.weight, m.bias, old_specs, new_specs)

        def _forward_linear_with_sharding(m,
                                          *args,
                                          old_specs=None,
                                          new_specs=None):
            out = _forward_linear(m,
                                  *args,
                                  old_specs=old_specs,
                                  new_specs=new_specs)
            # Add some hints to SPMD for sharding propagating
            context.tp_mark_sharding(out, ("dp", None, "tp"))
            return out

        assert os.environ.get("ACC_LLAMA_MLP") != "1"
        dp_dim = "dp" if (self.args.use_zero3
                          or self.args.fsdp_num > 1) else None
        for name, m in model.named_modules():
            # attn
            if "q_proj" in name:
                context.tp_mark_sharding(m.weight, ("tp", dp_dim))
            if "k_proj" in name:
                context.tp_mark_sharding(m.weight, ("tp", dp_dim))
            if "v_proj" in name:
                context.tp_mark_sharding(m.weight, ("tp", dp_dim))
            if "o_proj" in name:
                context.tp_mark_sharding(m.weight, (dp_dim, "tp"))
            # mlp
            if "gate_proj" in name:
                context.tp_mark_sharding(m.weight, ("tp", dp_dim))
            if "up_proj" in name:
                context.tp_mark_sharding(m.weight, ("tp", dp_dim))
            if "down_proj" in name:
                context.tp_mark_sharding(m.weight, (dp_dim, "tp"))

            # attn linear
            if self.args.use_zero2 or self.args.use_zero3 or self.args.fsdp_num > 1:
                tp_dp_linear = functools.partial(_forward_linear,
                                                 old_specs=("tp", None),
                                                 new_specs=("tp", "dp"))
                dp_tp_linear = functools.partial(_forward_linear,
                                                 old_specs=(None, "tp"),
                                                 new_specs=("dp", "tp"))
                tp_dp_linear_with_shard = functools.partial(
                    _forward_linear_with_sharding,
                    old_specs=("tp", None),
                    new_specs=("tp", "dp"))
            else:
                tp_dp_linear = _forward_linear
                dp_tp_linear = _forward_linear
                tp_dp_linear_with_shard = _forward_linear_with_sharding
            if "q_proj" in name:
                m.forward = MethodType(tp_dp_linear, m)
            if "k_proj" in name:
                m.forward = MethodType(tp_dp_linear, m)
            if "v_proj" in name:
                m.forward = MethodType(tp_dp_linear, m)
            if "o_proj" in name:
                m.forward = MethodType(dp_tp_linear, m)

            # mlp linear
            if "gate_proj" in name:
                m.forward = MethodType(tp_dp_linear_with_shard, m)
            if "up_proj" in name:
                m.forward = MethodType(tp_dp_linear_with_shard, m)
            if "down_proj" in name:
                m.forward = MethodType(dp_tp_linear, m)

    def tensor_parallel(self, model):
        def _grad_ag(grad):
            # insert all-gather
            xm.optimization_barrier_([grad])
            grad = grad.view(grad.size())
            mark_sharding(grad, self.tp_mesh, (0, None, 2))
            return grad

        def _forward_linear(m, *args):
            h = args[0]
            if self.args.sp_reshard_after_forward:
                return tensor_parallel.PatchedLinearForSP.apply(
                    h, m.weight, m.bias, False, self.tp_mesh)
            return torch.einsum("bij,jk->bik", h, m.weight.T)

        def _forward_sp_linear(m, *args):
            h = args[0]
            if self.args.sp_reshard_after_forward:
                return tensor_parallel.PatchedLinearForSP.apply(
                    h, m.weight, m.bias, True, self.tp_mesh)
            return torch.einsum("bij,jk->bik", h, m.weight.T)

        def _forward_ag_sp(m, *args, **kwargs):
            h = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            if not self.args.sp_reshard_after_forward:
                # manually insert all gather before sequence parallel
                xm.optimization_barrier_([h])
                h = h.view(h.size())
                mark_sharding(h, self.tp_mesh, (0, None, None))

            if len(args) > 0:
                as_list = list(args)
                as_list[0] = h
                args = tuple(as_list)
            else:
                kwargs["hidden_states"] = h
            output = m._original_forward(*args, **kwargs)
            out_h = output[0] if isinstance(output, tuple) else output
            # shard on sequence dim.
            mark_sharding(out_h, self.tp_mesh, (0, 1, 2))
            if out_h.requires_grad:
                out_h.register_hook(lambda grad: _grad_ag(grad))
            return output

        row_dim = 0 if (self.args.use_zero3
                        or self.args.fsdp_num > 1) else None
        col_dim = 1 if self.args.use_zero3 or self.args.fsdp_num > 1 else None
        # TODO: 切分Embedding，判断显存是否大
        device = lazy_device()
        gc_cnt = self.args.gc_cnt
        for decoder_layer in model.model.layers:
            is_torchdistX_deferred_init = (LOW_CPU_MEM_USAGE and any(
                fake.is_fake(param) for param in decoder_layer.parameters()))
            if is_torchdistX_deferred_init:
                deferred_init.materialize_module(decoder_layer)
            decoder_layer.to(device)
            # attn
            if hasattr(decoder_layer.self_attn, "_create_tp_mesh"):
                dp_num = self.args.dp_num
                if self.args.fsdp_num > 1:
                    dp_num = self.args.fsdp_num
                decoder_layer.self_attn._create_tp_mesh(
                    self.args.tp_num, dp_num)
            mark_sharding(decoder_layer.self_attn.q_proj.weight,
                          self.tp_row_mesh, (0, col_dim))
            mark_sharding(decoder_layer.self_attn.k_proj.weight,
                          self.tp_row_mesh, (0, col_dim))
            mark_sharding(decoder_layer.self_attn.v_proj.weight,
                          self.tp_row_mesh, (0, col_dim))
            mark_sharding(decoder_layer.self_attn.o_proj.weight,
                          self.tp_col_mesh, (row_dim, 1))
            # mlp
            if not os.environ.get("ACC_LLAMA_MLP") == "1":
                mark_sharding(decoder_layer.mlp.gate_proj.weight,
                              self.tp_row_mesh, (0, col_dim))
                mark_sharding(decoder_layer.mlp.up_proj.weight,
                              self.tp_row_mesh, (0, col_dim))
            else:
                decoder_layer.mlp.new_up_proj.weight.data = \
                    torch.stack([decoder_layer.mlp.gate_proj.weight.data.T,
                                 decoder_layer.mlp.up_proj.weight.data.T], dim=2)
                mark_sharding(decoder_layer.mlp.new_up_proj.weight,
                              self.tp_mesh, (row_dim, 1, None))
                del decoder_layer.mlp.gate_proj
                del decoder_layer.mlp.up_proj
                decoder_layer.mlp.gate_proj = None
                decoder_layer.mlp.up_proj = None
            mark_sharding(decoder_layer.mlp.down_proj.weight, self.tp_col_mesh,
                          (row_dim, 1))

            # sequence parallelism for LayerNorm
            if self.args.sp:
                # To combine all-reduce and sequence split into a reduce-scatter Op,
                # we use einsum to instead of origianl linear to ensure that there is
                # no other Ops between all-reduce and sequence sharding (as required
                # from reduce_scatter_utils).

                # attn linear
                decoder_layer.self_attn.q_proj.forward = \
                    MethodType(_forward_sp_linear, decoder_layer.self_attn.q_proj)
                decoder_layer.self_attn.k_proj.forward = \
                    MethodType(_forward_sp_linear, decoder_layer.self_attn.k_proj)
                decoder_layer.self_attn.v_proj.forward = \
                    MethodType(_forward_sp_linear, decoder_layer.self_attn.v_proj)
                decoder_layer.self_attn.o_proj.forward = \
                    MethodType(_forward_linear, decoder_layer.self_attn.o_proj)
                # mlp linear
                if not os.environ.get("ACC_LLAMA_MLP") == "1":
                    decoder_layer.mlp.gate_proj.forward = \
                        MethodType(_forward_sp_linear, decoder_layer.mlp.gate_proj)
                    decoder_layer.mlp.up_proj.forward = \
                        MethodType(_forward_sp_linear, decoder_layer.mlp.up_proj)
                decoder_layer.mlp.down_proj.forward = \
                    MethodType(_forward_linear, decoder_layer.mlp.down_proj)

                # insert all-gather and reduce-scatter into input and output of
                # self_attn and mlp.
                decoder_layer.self_attn._original_forward = \
                    decoder_layer.self_attn.forward
                decoder_layer.self_attn.forward = \
                    MethodType(_forward_ag_sp, decoder_layer.self_attn)

                decoder_layer.mlp._original_forward = \
                    decoder_layer.mlp.forward
                decoder_layer.mlp.forward = \
                    MethodType(_forward_ag_sp, decoder_layer.mlp)
            if self.args.gc:
                if gc_cnt > 0:
                    decoder_layer = checkpoint_module(decoder_layer)
                    gc_cnt -= 1

        is_torchdistX_deferred_init = (LOW_CPU_MEM_USAGE and any(
            fake.is_fake(param) for param in model.parameters()))
        if is_torchdistX_deferred_init:
            deferred_init.materialize_module(
                model,
                check_fn=lambda k: not isinstance(k, type(model.model.layers[0]
                                                          )))
        model.to(device)
        return model

    def shard_grad(self, model):
        for decoder_layer in model.model.layers:
            mark_sharding(decoder_layer.self_attn.q_proj.weight.grad,
                          self.tp_row_mesh, (0, 1))
            mark_sharding(decoder_layer.self_attn.k_proj.weight.grad,
                          self.tp_row_mesh, (0, 1))
            mark_sharding(decoder_layer.self_attn.v_proj.weight.grad,
                          self.tp_row_mesh, (0, 1))
            mark_sharding(decoder_layer.self_attn.o_proj.weight.grad,
                          self.tp_col_mesh, (0, 1))
            if not os.environ.get("ACC_LLAMA_MLP") == "1":
                mark_sharding(decoder_layer.mlp.gate_proj.weight.grad,
                              self.tp_row_mesh, (0, 1))
                mark_sharding(decoder_layer.mlp.up_proj.weight.grad,
                              self.tp_row_mesh, (0, 1))
            else:
                mark_sharding(decoder_layer.mlp.new_up_proj.weight.grad,
                              self.tp_mesh, (0, 1, None))
            mark_sharding(decoder_layer.mlp.down_proj.weight.grad,
                          self.tp_col_mesh, (0, 1))
        return model


AcceleratorFactory.regist("acc-llama", ACCLLAMAAccelerator)
