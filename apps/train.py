import torch

def train():
    torch.manual_seed(101)

    # parse args
    from flashmodels import arguments
    args = arguments.parse()

    # build model, tokenizer, loader, optimizer and lr_scheduler
    # and use accelerator to speed up training
    from flashmodels import Builder, Trainer, accelerate
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
