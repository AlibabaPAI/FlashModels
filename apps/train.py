import torch

from flashmodels import Builder, Trainer, accelerate, arguments


def train():
    torch.manual_seed(101)

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
