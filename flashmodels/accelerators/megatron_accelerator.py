from flashmodels.accelerators.accelerator import (Accelerator,
                                                  AcceleratorFactory)


class MegatronAccelerator(Accelerator):
    def accelerate(self, model, loader):
        raise NotImplemented("MegatronAccelerator is not implemented.")


AcceleratorFactory.regist("megatron", MegatronAccelerator)
