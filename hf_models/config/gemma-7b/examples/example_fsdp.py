# Make sure to run the script with the following envs:
#   PJRT_DEVICE=TPU XLA_USE_SPMD=1

import torch
import torch_xla

import torch_xla.core.xla_model as xm

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# Set up TPU device.
device = xm.xla_device()
model_id = "google/gemma-7b"

# Load the pretrained model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# Set up PEFT LoRA for fine-tuning.
lora_config = LoraConfig(
    r=8,
    target_modules=["k_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# Load the dataset and format it for training.
data = load_dataset("Abirate/english_quotes", split="train")
max_seq_length = 1024

# Set up the FSDP config. To enable FSDP via SPMD, set xla_fsdp_v2 to True.
fsdp_config = {"fsdp_transformer_layer_cls_to_wrap": [
        "GemmaDecoderLayer"
    ],
    "xla": True,
    "xla_fsdp_v2": True,
    "xla_fsdp_grad_ckpt": True}

# Finally, set up the trainer and train the model.
trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    args=TrainingArguments(
        per_device_train_batch_size=64,  # This is actually the global batch size for SPMD.
        num_train_epochs=100,
        max_steps=-1,
        output_dir="./output",
        optim="adafactor",
        logging_steps=1,
        dataloader_drop_last = True,  # Required for SPMD.
        fsdp="full_shard",
        fsdp_config=fsdp_config,
    ),
    peft_config=lora_config,
    dataset_text_field="quote",
    max_seq_length=max_seq_length,
    packing=True,
)

trainer.train()