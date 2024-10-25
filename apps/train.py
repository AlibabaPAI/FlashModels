import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm

from flashmodels import Builder, Trainer, accelerate, arguments


def train():
    seed = 101
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    xm.set_rng_state(seed)
    torch_xla._XLAC._xla_set_use_full_mat_mul_precision(
        use_full_mat_mul_precision=True)

    # parse args
    args = arguments.parse()

    # build model, tokenizer, loader, optimizer and lr_scheduler
    # and use accelerator to speed up training
    builder = Builder(args)
    model, loader, tokenizer = builder.build_model_dataloader()
    model, loader = accelerate(model, loader, args)
    optimizer, lr_scheduler = builder.build_optimizer_scheduler(
        model, loader, args)

    # train jobs
    trainer = Trainer(model, loader, optimizer, lr_scheduler, tokenizer, args)
    trainer.train()


if __name__ == "__main__":
    train()
