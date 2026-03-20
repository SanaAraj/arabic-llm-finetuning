import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import (
    MODEL_ID, FALLBACK_MODEL_ID,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES
)

HAS_CUDA = torch.cuda.is_available()


def get_quantization_config():
    if not HAS_CUDA:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config():
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(model_id=None, use_lora=True):
    model_id = model_id or MODEL_ID
    bnb_config = get_quantization_config()

    load_kwargs = {"trust_remote_code": True}
    if bnb_config:
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["dtype"] = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
        print(f"Falling back to {FALLBACK_MODEL_ID}")
        model = AutoModelForCausalLM.from_pretrained(FALLBACK_MODEL_ID, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_ID, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_lora:
        if bnb_config:
            model = prepare_model_for_kbit_training(model)
        lora_config = get_lora_config()
        model = get_peft_model(model, lora_config)

    return model, tokenizer


def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    print(f"Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")


if __name__ == "__main__":
    print(f"CUDA available: {HAS_CUDA}")
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(use_lora=True)
    print_trainable_parameters(model)
