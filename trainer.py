import torch
import os
import json
import numpy as np
from datasets import Dataset
import pandas as pd
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
import transformers
# from awq import AutoAWQForCausalLM
from datetime import datetime
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
DATASET_PATH = "bitAgent.csv"
OUTPUT_DIR = "/home/user/saud/models/fine-tuned-mistral-bitagent-latest"
QUANT_DIR = "/home/user/saud/models/fine-tuned-mistral-bitagent-quantized"
HF_TOKEN = os.getenv("HF_TOKEN")

# --- LOGIN TO HUGGINGFACE ---
login(token=HF_TOKEN)

# --- LOAD AND PREPARE DATASET ---
df = pd.read_csv(DATASET_PATH)
df = df.rename(columns={"input": "user", "output": "assistant"})
dataset = Dataset.from_pandas(df)
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)

# --- MODEL AND TOKENIZER SETUP ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

# --- DATA PREPROCESSING ---
# Add this before preprocessing to analyze your data
def analyze_lengths(dataset):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    lengths = []
    for example in dataset:
        text = f"[INST] {example['user']} [/INST]\n{example['assistant']}</s>"
        tokens = tokenizer(text)["input_ids"]
        lengths.append(len(tokens))
    
    print(f"Average length: {np.mean(lengths):.1f}")
    print(f"95th percentile: {np.percentile(lengths, 95):.1f}")
    print(f"Max length: {max(lengths)}")

# analyze_lengths(dataset)

def preprocess_function(example):
    # Format with Mistral's chat template
    text = f"[INST] {example['user']} [/INST]\n{example['assistant']}</s>"
    
    # Tokenize
    tokens = tokenizer(
        text,
        max_length=768,
        truncation=True,
        padding="max_length",
    )
    
    # Create labels mask (only calculate loss on assistant response)
    input_part = tokenizer.encode(
        f"[INST] {example['user']} [/INST]\n", 
        add_special_tokens=False,
        max_length=768,
        truncation=True
    )
    labels = [-100] * len(input_part) + tokens.input_ids[len(input_part):]
    
    return {
        "input_ids": tokens.input_ids,
        "attention_mask": tokens.attention_mask,
        "labels": labels
    }

tokenized_train = train_test_split["train"].map(preprocess_function)
tokenized_val = train_test_split["test"].map(preprocess_function)

# --- LORA CONFIGURATION ---
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# --- METRICS CALCULATION ---
def compute_metrics(p):
    logits = p.predictions
    labels = p.label_ids
    
    # Replace -100 with pad token id
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    
    # Decode predictions and labels
    preds = np.argmax(logits, axis=-1)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Extract JSON parts
    def extract_json(text):
        try:
            return json.loads(text.split("[/INST]\n")[-1].strip())
        except:
            return {}
    
    pred_jsons = [extract_json(p) for p in decoded_preds]
    label_jsons = [extract_json(l) for l in decoded_labels]
    
    # Calculate accuracy
    acc = sum([1 for p, l in zip(pred_jsons, label_jsons) if p == l])/len(pred_jsons)
    
    return {"accuracy": acc}

# --- TRAINING SETUP ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=1e-5,
    bf16=True,
    optim="paged_adamw_32bit",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_ratio=0.1,
    max_grad_norm=0.3,
    report_to="none",
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    compute_metrics=compute_metrics,
)

# --- SAVE MODEL ---
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
# Load the configuration from the fine-tuned model and save it to the same directory
config = AutoConfig.from_pretrained(OUTPUT_DIR)
config.save_pretrained(OUTPUT_DIR)

# --- QUANTIZE ---
# Load model for AWQ quantization
# awq_model = AutoAWQForCausalLM.from_pretrained(OUTPUT_DIR)

# # Define quantization config (adjust parameters as needed)
# quant_config = {
#     "zero_point": True,  # if supported by your AWQ version
#     # "q_group_size": 128,
#     "w_bit": 4,  # often required for weight bit-width
# }

# Apply quantization
# awq_model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model to a separate directory

# awq_model.save_quantized(QUANT_DIR)
# tokenizer.save_pretrained(QUANT_DIR)  # save tokenizer for the quantized model

print(f"Original model saved to {OUTPUT_DIR}")
# print(f"Quantized model saved to {QUANT_DIR}")
# print("Quantized directory contents:", os.listdir(QUANT_DIR))
print(os.listdir(OUTPUT_DIR))

# --- EVALUATION ---
generation_config = model.generation_config
generation_config.max_new_tokens = 256
generation_config.temperature = 0.01
generation_config.top_p = 0.95
generation_config.repetition_penalty = 1.15

eval_prompt = "Distance from Los Angeles to New York"
model_input = tokenizer(
    f"[INST] {eval_prompt} [/INST]\n", 
    return_tensors="pt"
).to("cuda")

ft_model = PeftModel.from_pretrained(model, OUTPUT_DIR)
ft_model.eval()

with torch.no_grad():
    output = ft_model.generate(
        **model_input,
        generation_config=generation_config
    )
    
print(tokenizer.decode(output[0], skip_special_tokens=True))