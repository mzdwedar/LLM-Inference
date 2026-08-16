import torch
from datasets import load_dataset  # Make sure to: pip install datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from compressed_tensors.quantization import QuantizationArgs
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# 1. Load the model in BF16 (strongly recommended over FP32)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2. Add calibration dataset (required for KV-cache and activation profiling)
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset(DATASET_ID, split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")


def preprocess_and_tokenize(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(preprocess_and_tokenize, remove_columns=ds.column_names)

# 3. Configure the combined recipe: Weights (Static FP8) + Activations (Dynamic FP8) + KV Cache (FP8 Per-Head)
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",  # Static weights + dynamic activation quantization
    ignore=["lm_head"],
    kv_cache_scheme=QuantizationArgs(
        num_bits=8,
        type="float",
        strategy="attn_head",  # "attn_head" preserves perplexity better than "tensor"
    ),
)

# 4. Run the Calibration Pipeline
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# 5. Save the final FP8 W8A8 + KV model
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-W8A8-KV"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Successfully saved fully quantized model to {SAVE_DIR}")