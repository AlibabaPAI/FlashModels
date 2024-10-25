import itertools
import logging

import datasets
import torch
import transformers


def get_hf_dataset_loader(tokenizer, args):
    if args.dataset_name_or_path.endswith(".json"):
        raw_datasets = datasets.load_dataset(
            "json", data_files=args.dataset_name_or_path)
    else:
        raw_datasets = datasets.load_dataset(args.dataset_name_or_path,
                                             args.dataset_config)

    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name],
                         return_token_type_ids=False)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
    )

    block_size = args.max_seq_length

    def group_texts(examples):
        concatenated_examples = {
            k: list(itertools.chain(*examples[k]))
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k:
            [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )

    train_dataset = lm_datasets["train"]
    # DataLoader creation
    train_sampler = None
    data_num_replicas = args.fsdp_num * args.dp_num / args.sp_num
    # data_num_replicas = args.fsdp_num * args.dp_num
    
    if args.pp_num > 1:
        # disable sampler for now
        # the rank below should be:
        # config.get_mesh().get_dp_rank() * config.get_mesh().get_fsdp_num() \
        # + config.get_mesh().get_fsdp_rank()
        args.disable_train_sampler = True
    if (not args.disable_train_sampler) and (data_num_replicas > 1) \
            and (not args.tp_num > 1) and (not args.spmd_fsdp) and (not args.sp_num > 1):
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=(1 if args.tp_num > 1 else data_num_replicas),
            rank=(0 if args.tp_num > 1 else args.global_rank),
            shuffle=True)

    bs = args.micro_batch_size
    if args.tp_num > 1:
        bs *= args.dp_num
    if args.pp_num > 1:
        bs *= args.gradient_accumulation_steps
    if args.spmd_fsdp:
        bs *= int(args.fsdp_num / args.sp_num)
        # bs *= args.fsdp_num
    print(f"{bs=}")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=bs,
        collate_fn=transformers.default_data_collator,
        sampler=train_sampler,
        drop_last=True)
    return train_dataloader
