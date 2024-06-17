import torchacc as ta

from flashmodels.accelerators.accelerator import (Accelerator,
                                                  AcceleratorFactory)


class ACCGemmaAccelerator(Accelerator):

    def accelerate(self, model, loader):
        model, loader = self.accelerate_internal(model, loader)
        return model, loader

    def accelerate_internal(self, model, loader):
        config = self.get_config()
        model = ta.accelerate(model, config)
        return model, loader

    def get_config(self):
        config = ta.Config()
        config.compute.fp16 = self.args.fp16
        config.compute.bf16 = self.args.bf16

        config.memory.gc = self.args.gc
        if self.args.gc:
            config.memory.gc_cls = {"GemmaDecoderLayer"}

        config.dist.fsdp.size = self.args.fsdp_num
        config.dist.fsdp.wrap_layer_cls = {"GemmaDecoderLayer"}
        config.dist.fsdp.flatten_parameters = True

        return config


AcceleratorFactory.regist('acc-gemma', ACCGemmaAccelerator)
