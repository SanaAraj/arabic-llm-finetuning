import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import MODEL_ID, OUTPUT_DIR
from model import HAS_CUDA


def load_model_for_inference(use_adapter=True):
    load_kwargs = {"trust_remote_code": True}
    if not HAS_CUDA:
        load_kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    if use_adapter and os.path.exists(OUTPUT_DIR):
        model = PeftModel.from_pretrained(model, OUTPUT_DIR)
        print(f"Loaded adapter from {OUTPUT_DIR}")
    else:
        print("Using base model without adapter")

    return model, tokenizer


def format_prompt(instruction):
    return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"


def generate(model, tokenizer, prompt, max_new_tokens=256):
    formatted = format_prompt(prompt)
    inputs = tokenizer(formatted, return_tensors="pt")

    if HAS_CUDA:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract just the assistant's response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
    return response.strip()


def run_inference(prompt, use_adapter=True):
    model, tokenizer = load_model_for_inference(use_adapter=use_adapter)
    response = generate(model, tokenizer, prompt)
    print(f"\nPrompt: {prompt}")
    print(f"\nResponse: {response}")
    return response


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--no_adapter", action="store_true")
    args = parser.parse_args()
    run_inference(args.prompt, use_adapter=not args.no_adapter)
