import torchacc as ta
from torchacc.dist.tp import mark_sharding

from flashmodels.accelerators.accelerator import (Accelerator,
                                                  AcceleratorFactory)


class ACCGemmaAccelerator(Accelerator):
    def accelerate(self, model, loader):
        model, loader = self.accelerate_internal(model, loader)
        return model, loader

    def accelerate_internal(self, model, loader):
        config = self.get_config()
        model = ta.accelerate(model, config=config)
        return model, loader

    def get_config(self):
        def _shard_output_callable(output, mesh):
            if not isinstance(output, tuple) and output['logits'] is not None:
                mark_sharding(output['logits'], mesh, ('fsdp', None, None))

        config = ta.Config()
        config.compute.fp16 = self.args.fp16
        config.compute.bf16 = self.args.bf16

        config.memory.gc = self.args.gc
        if self.args.gc:
            config.memory.gc_cls = {"GemmaDecoderLayer"}

        config.dist.fsdp.size = self.args.fsdp_num
        config.dist.fsdp.wrap_layer_cls = {"GemmaDecoderLayer"}
        config.dist.fsdp.flatten_parameters = True
        config.dist.fsdp.use_spmd = self.args.spmd_fsdp
        config.dist.fsdp.shard_output_callable = _shard_output_callable

        return config


AcceleratorFactory.regist('acc-gemma', ACCGemmaAccelerator)
