import gc
import os
import logging
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format
from lama_orpo.flash import setup_flash_attention
# =======================================================



attn_implementation, torch_dtype = setup_flash_attention()

logging.basicConfig(
    filename="main_output.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

attn_implementation, torch_dtype = setup_flash_attention()

logging.info(f"Attention backend: {attn_implementation}")
logging.info(f"Using dtype: {torch_dtype}")


base_model = "meta-llama/Meta-Llama-3-8B"
# base_model = "meta-llama/Llama-3.2-1B-Instruct"
new_model = "OrpoLlama-3-8B"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)

# Resetting Chat Template to avoid errors
tokenizer.chat_template = None
model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)

dataset_name = "mlabonne/orpo-dpo-mix-40k"
dataset = load_dataset(dataset_name, split="all")
dataset = dataset.shuffle(seed=42).select(range(100))

def format_chat_template(row):
    row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc= os.cpu_count(),
)
dataset = dataset.train_test_split(test_size=0.01)

from lama_orpo.orpo import orpo_trainer
trainer = orpo_trainer(model=model, dataset=dataset, tokenizer=tokenizer, peft_config=peft_config)
trainer.train()
trainer.save_model(new_model)


# Flush memory
del trainer, model
gc.collect()
torch.cuda.empty_cache()


# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model, tokenizer = setup_chat_format(model, tokenizer)

# Merge adapter with base model
model = PeftModel.from_pretrained(model, new_model)
model = model.merge_and_unload()

model.push_to_hub(new_model)
tokenizer.push_to_hub(new_model)

print("finished everything successfully...")