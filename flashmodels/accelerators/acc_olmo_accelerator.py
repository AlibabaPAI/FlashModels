import torchacc as ta
from torchacc.dist.tp import mark_sharding

from flashmodels.accelerators.accelerator import (Accelerator,
                                                  AcceleratorFactory)


class ACCOlmoAccelerator(Accelerator):
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
        else:
            raise NotImplementedError("Currently, only FSDP is supported.")

    def get_config(self, model):
        def _shard_output_callable(output, mesh):
            if not isinstance(output, tuple) and output['logits'] is not None:
                mark_sharding(output['logits'], mesh, ('fsdp', None, None))

        config = ta.Config()
        config.compute.fp16 = self.args.fp16
        config.compute.bf16 = self.args.bf16
        config.compute.acc_scaled_dot_attn = True

        config.memory.gc = self.args.gc
        if self.args.gc:
            config.memory.gc_cls = {"OlmoSequentialBlock"}

        config.dist.fsdp.size = self.args.fsdp_num
        config.dist.fsdp.wrap_layer_cls = {"OlmoSequentialBlock"}
        config.dist.fsdp.flatten_parameters = True
        config.dist.fsdp.use_spmd = self.args.spmd_fsdp
        config.dist.fsdp.shard_output_callable = _shard_output_callable

        return config


AcceleratorFactory.regist("acc-olmo", ACCOlmoAccelerator)
