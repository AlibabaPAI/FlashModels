# This file is mostly referenced from https://github.com/tatsu-lab/stanford_alpaca

#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
import os
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset

from flashmodels.logger import logger
from flashmodels.utils import jload

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input":
    ("Below is an instruction that describes a task, paired with an input that provides further context. "
     "Write a response that appropriately completes the request.\n\n"
     "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
     ),
    "prompt_no_input":
    ("Below is an instruction that describes a task. "
     "Write a response that appropriately completes the request.\n\n"
     "### Instruction:\n{instruction}\n\n### Response:"),
}


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if bool(int(os.environ.get("LOW_CPU_MEM_USAGE", "0"))):
        logger.warning(
            f"When set LOW_CPU_MEM_USAGE, alpaca data is not recommended,"
            f"as tokenizer and embedding will not been resized.")
        return
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer,
                 padding_strategy: str) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=padding_strategy,
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources: Sequence[str], targets: Sequence[str],
               tokenizer: transformers.PreTrainedTokenizer,
               padding_strategy: str) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer, padding_strategy)
        for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 padding_strategy: str):
        super(SupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT[
            "prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != ""
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}"
            for example in list_data_dict
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer, padding_strategy)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id).int(),
        )


def get_alpaca_loader(model, tokenizer, args):
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    tokenizer.model_max_length = args.max_seq_length
    train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                      data_path=args.dataset_name_or_path,
                                      padding_strategy=args.padding_strategy)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    train_sampler = None
    data_num_replicas = args.fsdp_num * args.dp_num
    if args.pp_num > 1:
        # disable sampler for now
        # the rank below should be:
        # config.get_mesh().get_dp_rank() * config.get_mesh().get_fsdp_num() \
        # + config.get_mesh().get_fsdp_rank()
        args.disable_train_sampler = True
    if (not args.disable_train_sampler) and (data_num_replicas > 1) \
            and (not args.tp_num > 1) and (not args.spmd_fsdp):
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=data_num_replicas,
            rank=args.global_rank % data_num_replicas,
            seed=42)

    bs = args.micro_batch_size
    if args.tp_num > 1:
        bs *= args.dp_num
    if args.pp_num > 1:
        bs *= args.gradient_accumulation_steps
    if args.spmd_fsdp:
        bs *= args.fsdp_num
    loader = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=bs,
                                         collate_fn=data_collator,
                                         sampler=train_sampler,
                                         drop_last=True)

    return loader
