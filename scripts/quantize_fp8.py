import os
import shutil

# Ensure Hugging Face downloads and temp files stay in /workspace
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_HUB_CACHE"] = "/workspace/hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"  # Fallback to standard HTTP download stream

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAVE_DIR = f"/workspace/{MODEL_ID.split('/')[-1]}-FP8-Dynamic"

print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2. Configure dynamic FP8 quantization scheme
recipe = QuantizationModifier(
    targets="Linear", 
    scheme="FP8_DYNAMIC", 
    ignore=["lm_head"]
)

print("Applying FP8 dynamic quantization recipe...")
oneshot(model=model, recipe=recipe)

# 3. Save quantized model and tokenizer directly into /workspace
print(f"Saving quantized weights and configs to: {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"✅ Quantization complete! Ready to serve from {SAVE_DIR}")