from datasets import load_dataset
from config import DATASET_ID


def format_instruction(sample):
    """Format sample into Llama 3.1 Instruct chat template."""
    text = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{sample['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{sample['output']}<|eot_id|>"
    )
    return {"text": text}


def load_and_prepare_data(test_size=0.1):
    dataset = load_dataset(DATASET_ID)

    # CIDAR has train split only, we split it ourselves
    train_data = dataset["train"]
    split = train_data.train_test_split(test_size=test_size, seed=42)

    train_formatted = split["train"].map(format_instruction)
    eval_formatted = split["test"].map(format_instruction)

    return train_formatted, eval_formatted


def preview_data(n=3):
    train_data, eval_data = load_and_prepare_data()

    print(f"Train samples: {len(train_data)}")
    print(f"Eval samples: {len(eval_data)}")
    print("\n" + "="*60 + "\n")

    for i, sample in enumerate(train_data.select(range(n))):
        print(f"--- Sample {i+1} ---")
        print(sample["text"])
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    preview_data()
