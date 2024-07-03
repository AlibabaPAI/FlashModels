from abc import ABC, abstractmethod

_accelerators = {}


class Accelerator(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def accelerate(self):
        pass

    def shard_grad(self, model):
        """ shard gradient for zero2 or zero3
        """
        raise NotImplementedError


class AcceleratorFactory:
    @staticmethod
    def regist(key, builder):
        global _accelerators
        _accelerators[key] = builder

    @staticmethod
    def get(key, args):
        global _accelerators
        acc_name = key + "-" + args.model_type if args.model_type else key
        accelerator = _accelerators.get(acc_name)
        if not accelerator:
            raise ValueError(f"Unkown accelerator: {acc_name}")
        return accelerator(args)
