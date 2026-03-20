# Arabic LLM Fine-tuning Pipeline

A QLoRA fine-tuning pipeline for Arabic instruction-following language models.

## Overview

This project fine-tunes a pre-trained language model on Arabic instruction data using QLoRA (Quantized Low-Rank Adaptation). QLoRA enables efficient fine-tuning by quantizing the base model to 4-bit precision and training only a small set of adapter parameters, making it possible to fine-tune large models on consumer hardware.

## Dataset

The pipeline uses [CIDAR](https://huggingface.co/datasets/arbml/CIDAR) (Arabic Instruction Dataset), a collection of 10,000 Arabic instruction-response pairs. CIDAR covers diverse tasks including creative writing, question answering, summarization, and general knowledge.

## Model

By default, the pipeline uses Qwen2.5-0.5B-Instruct for quick local testing. For production use, you can configure it to use larger models like Llama 3.1 8B Instruct with proper Hugging Face authentication.

## Setup

```bash
git clone https://github.com/SanaAraj/arabic-llm-finetuning.git
cd arabic-llm-finetuning
pip install -r requirements.txt
```

For gated models (e.g., Llama), log in to Hugging Face:
```bash
huggingface-cli login
```

## Usage

Preview the dataset:
```bash
python main.py --prepare
```

Train the model:
```bash
python main.py --train --max_steps 100 --batch_size 4 --lr 2e-4
```

Run inference:
```bash
python main.py --infer --prompt "ما هي عاصمة مصر؟"
```

Evaluate on held-out samples:
```bash
python main.py --eval --n_eval 5
```

## Hardware Requirements

- GPU with 12-24GB VRAM recommended for full training
- CPU-only works but is slow (model loads without quantization)
- Compatible with Google Colab free tier using QLoRA

To run on Colab, install dependencies and run the training script with a small number of steps to verify everything works.

## Tech Stack

- transformers
- peft
- bitsandbytes
- datasets
- trl
- accelerate

## Training Configuration

| Parameter | Value |
|-----------|-------|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Learning rate | 2e-4 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Max sequence length | 512 |

LoRA adapters are saved to the `outputs/` directory. Only the adapter weights are saved, not the full model.

## Project Structure

```
├── main.py          # CLI entry point
├── prepare_data.py  # Dataset loading and formatting
├── model.py         # Model loading with QLoRA config
├── train.py         # Training loop using SFTTrainer
├── infer.py         # Inference script
├── evaluate.py      # Qualitative evaluation
├── config.py        # Shared configuration
├── requirements.txt
└── .gitignore
```
