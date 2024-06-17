from flashmodels.datasets.alpaca import get_alpaca_loader
from flashmodels.datasets.hf_dataset import get_hf_dataset_loader


def get_dataloader(model, tokenizer, args):
    if "alpaca" in args.dataset_name_or_path:
        return get_alpaca_loader(model, tokenizer, args)
    else:
        return get_hf_dataset_loader(tokenizer, args)
