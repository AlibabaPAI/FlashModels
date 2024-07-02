import numpy as np
import torchacc as ta
from torchacc import lazy_device
from torchacc.dist.tp import mark_sharding

from flashmodels.accelerators.accelerator import (Accelerator,
                                                  AcceleratorFactory)


class ACCGPTAccelerator(Accelerator):
    def accelerate(self, model, loader):
        model, loader = self.accelerate_internal(model, loader)

        return model, loader

    def accelerate_internal(self, model, loader):
        if not (self.args.tp_num > 1 or self.args.sp_num > 1):
            if self.args.resume_from_checkpoint:
                raise NotImplementedError("resume_from_checkpoint.")

            config = self.get_config(model)
            model = ta.accelerate(model, config=config)
            return model, loader

        device = lazy_device()
        model.to(device)

        if self.args.tp_num > 1:
            model = self.tensor_parallel(model)

        return model, loader

    def get_config(self, model):
        def _shard_output_callable(output, mesh):
            if not isinstance(output, tuple) and output['logits'] is not None:
                mark_sharding(output['logits'], mesh, ('fsdp', None, None))

        config = ta.Config()
        config.compute.fp16 = self.args.fp16
        config.compute.bf16 = self.args.bf16

        config.memory.gc = self.args.gc
        if self.args.gc:
            config.memory.gc_cls = {"GPT2Block"}

        config.dist.fsdp.size = self.args.fsdp_num
        config.dist.fsdp.wrap_layer_cls = {"GPT2Block"}
        config.dist.fsdp.flatten_parameters = True
        config.dist.fsdp.use_spmd = self.args.spmd_fsdp
        config.dist.fsdp.shard_output_callable = _shard_output_callable

        return config

    def tensor_parallel(self, model):
        num_devices = self.args.tp_num
        mask_mesh_shape = (num_devices, 1)
        devices_ids = np.arange(num_devices)
        mask_mesh = Mesh(devices_ids, mask_mesh_shape, ("X", "Y"))
        label_mesh_shape = (num_devices, )
        label_mesh = Mesh(devices_ids, label_mesh_shape, ("X"))

        for encoder_layer in model.transformer.h:
            mark_sharding(encoder_layer.self_attn.q_proj.weight, mask_mesh,
                          (0, 1))
            mark_sharding(encoder_layer.self_attn.k_proj.weight, mask_mesh,
                          (0, 1))
            mark_sharding(encoder_layer.self_attn.v_proj.weight, mask_mesh,
                          (0, 1))
            mark_sharding(encoder_layer.self_attn.o_proj.weight, mask_mesh,
                          (0, 1))

            mark_sharding(encoder_layer.mlp.gate_proj.weight, mask_mesh,
                          (0, 1))
            mark_sharding(encoder_layer.mlp.down_proj.weight, mask_mesh,
                          (1, 0))
            mark_sharding(encoder_layer.mlp.up_proj.weight, mask_mesh, (0, 1))


AcceleratorFactory.regist("acc-gpt", ACCGPTAccelerator)
