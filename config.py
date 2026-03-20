MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
FALLBACK_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "arbml/CIDAR"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

MAX_SEQ_LENGTH = 512
OUTPUT_DIR = "outputs"
