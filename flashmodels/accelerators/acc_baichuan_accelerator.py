import torchacc as ta

from flashmodels.accelerators.accelerator import (Accelerator,
                                                  AcceleratorFactory)


class ACCBaiChuanAccelerator(Accelerator):

    def accelerate(self, model, loader):
        model, loader = self.accelerate_internal(model, loader)

        return model, loader

    def accelerate_internal(self, model, loader):
        if not (self.args.tp_num > 1 or self.args.sp_num > 1):
            if self.args.resume_from_checkpoint:
                raise NotImplementedError("resume_from_checkpoint.")

            config = self.get_config(model)
            model = ta.accelerate(model, config)
            return model, loader

    def get_config(self, model):
        config = ta.Config()
        config.compute.fp16 = self.args.fp16
        config.compute.bf16 = self.args.bf16

        config.memory.gc = self.args.gc
        if self.args.gc:
            config.memory.gc_cls = {"BaichuanLayer"}

        config.dist.fsdp.size = self.args.fsdp_num
        config.dist.fsdp.wrap_layer_cls = {"BaichuanLayer"}
        config.dist.fsdp.flatten_parameters = True

        return config


AcceleratorFactory.regist("acc-baichuan", ACCBaiChuanAccelerator)
