import os
from trl import SFTTrainer, SFTConfig
from model import load_model_and_tokenizer
from prepare_data import load_and_prepare_data
from config import MAX_SEQ_LENGTH, OUTPUT_DIR


def train(max_steps=100, batch_size=4, learning_rate=2e-4):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(use_lora=True)
    train_data, eval_data = load_and_prepare_data()

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        fp16=False,
        optim="adamw_torch",
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapters to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    train(max_steps=args.max_steps, batch_size=args.batch_size, learning_rate=args.lr)
