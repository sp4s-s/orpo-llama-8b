from trl import ORPOConfig, ORPOTrainer

def get_orpo_args():
    orpo_args = ORPOConfig(
        output_dir="./results/",
        overwrite_output_dir=True,

        # Core training
        learning_rate=8e-6,
        beta=0.1,
        lr_scheduler_type="linear",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        num_train_epochs=1,

        # Eval & Logging
        eval_strategy="steps",              
        # eval_steps=0.2,
        eval_accumulation_steps=2,
        logging_steps=1,
        report_to="wandb",

        # Lengths
        max_length=1024,
        max_prompt_length=512,

        # Optional: enable fp16/bfloat16
        bf16=True,  
        fp16=False,
        save_strategy="epoch",
    )

    return orpo_args

def orpo_trainer(model, dataset, tokenizer, peft_config):
    orpo_args = get_orpo_args()
    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,              # ✅ instead of tokenizer=...
        peft_config=peft_config.to_dict(),       # ✅ make sure it's a dict
    )
    return trainer



