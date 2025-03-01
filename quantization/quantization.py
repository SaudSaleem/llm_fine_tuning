from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
HF_USERNAME = os.getenv('HF_USERNAME')

login(token=HF_TOKEN)
MODEL_NAME = "saudsaleem/bitagent-autotrain"
QUANTIZED_MODEL_NAME = "bitagent-autotrain-GPTQ-8bit"
# Define the quantization configuration
quantize_config = BaseQuantizeConfig(
    bits=8,             # Use 8-bit quantization
    group_size=32,     # Group size for quantization (smaller = better compression)
    desc_act=True,      # Disable activation scaling for faster inference
    use_cuda_fp16=True,
    true_sequential=True
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load and quantize the model
model = AutoGPTQForCausalLM.from_pretrained(
    "saudsaleem/bitagent-autotrain",
    quantize_config=quantize_config,
    device_map="auto"   # Automatically distribute model across available GPUs
)

# Save the quantized model locally
model.save_quantized(QUANTIZED_MODEL_NAME)
tokenizer.save_pretrained(QUANTIZED_MODEL_NAME)

print("✅ Model successfully quantized and saved!")

# Push to Hugging Face Hub (optional)
model.push_to_hub(f"{HF_USERNAME}/{QUANTIZED_MODEL_NAME}")
tokenizer.push_to_hub(f"{HF_USERNAME}/{QUANTIZED_MODEL_NAME}")

print("✅ Quantized model pushed to the Hugging Face Hub!")
