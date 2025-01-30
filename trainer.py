import torch
import os
import json
import ast
import re
from typing import List, Dict, Any
from datasets import Dataset
from evaluate import load
import pandas as pd
import numpy as np
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EvalPrediction
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
import transformers

# --- CONFIGURATION ---
MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
DATASET_PATH = "bitAgent1.csv"
OUTPUT_DIR = "/home/user/saud/models/fine-tuned-mistral-bitagent-latest"
HF_TOKEN = os.getenv("HF_TOKEN")

# --- LOGIN TO HUGGINGFACE ---
login(token=HF_TOKEN)

# --- LOAD AND PREPROCESS DATASET ---
df = pd.read_csv(DATASET_PATH)

# Add system prompt to training data
def format_training_example(row):
    # system_prompt = """Respond ONLY with function call. Example:
    # User: What is the distance from Los Angeles to New York
    # Assistant: calculate_distance(destination="New York", origin="Los Angeles")"""
    
    return {
        "user": f"[INST] input: {row['input']} [/INST]",
        "assistant": f"{row['output']}</s>"
    }

# Create a list of formatted examples
formatted_df = df.apply(format_training_example, axis=1)

# Convert the formatted dataframe into a new DataFrame with 'user' and 'assistant' columns
formatted_df = pd.DataFrame(formatted_df.tolist())

# Then create the Dataset
dataset = Dataset.from_pandas(formatted_df)
train_test_split = dataset.train_test_split(test_size=0.15, seed=42)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

# --- DATA PROCESSING ---
# def preprocess_function(examples):
#     # print('saleem', examples['user'], examples['assistant'])
#     tokenized = tokenizer(
#         examples["user"],
#         text_target=examples["assistant"],
#         max_length=256,
#         truncation=True,
#         padding="max_length",
#     )
#     # Mask user input in labels
#     user_tokens = tokenizer(examples["user"], add_special_tokens=False)["input_ids"]
#     labels = [-100]*len(user_tokens) + tokenized["labels"][len(user_tokens):]

#     return {
#         "input_ids": tokenized["input_ids"],
#         "attention_mask": tokenized["attention_mask"],
#         "labels": labels
#     }
def preprocess_function(examples):
    # Tokenize user and assistant together with special tokens
    combined_texts = [user + assistant for user, assistant in zip(examples["user"], examples["assistant"])]
    tokenized = tokenizer(
        combined_texts,
        max_length=768,
        truncation=True,
        padding="max_length",
        add_special_tokens=True  # Ensures BOS/EOS are added
    )
    
    # Tokenize user alone to find actual length in combined tokens
    user_tokenized = tokenizer(examples["user"], add_special_tokens=False)
    user_lengths = [len(ids) for ids in user_tokenized["input_ids"]]
    
    # Account for BOS token added at start of combined text
    user_lengths = [len(tokenizer.encode(user, add_special_tokens=True)) - 1 for user in examples["user"]]
    
    # Create labels by masking user part (including BOS)
    labels = []
    for input_ids, user_len in zip(tokenized["input_ids"], user_lengths):
        label = [-100] * user_len + input_ids[user_len:]
        labels.append(label)
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }

tokenized_ds = train_test_split.map(
    preprocess_function,
    batched=True,
    batch_size=128
)

# --- LORA CONFIG ---
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
# model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    bias="none",
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
    modules_to_save=None
)
model = get_peft_model(model, peft_config)
print_trainable_parameters(model)
# --- TRAINING ARGS ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=2e-5,
    optim="paged_adamw_32bit",
    logging_steps=10,
    eval_strategy="epoch",
    eval_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="accuracy", 
    bf16=True,
    max_grad_norm=0.5,
    report_to="none",
    gradient_checkpointing=False,
)


# def compute_metrics(eval_pred: EvalPrediction):
#     # Get predictions and labels
#     predictions, labels = eval_pred
#     print('predictions shape', predictions.shape, 'labels shape', labels.shape)
#     # Convert logits to token IDs (assuming classification head)
#     predictions = np.argmax(predictions, axis=-1)
#     print('predictions shape after processing', predictions.shape)
#     # Remove ignored indices (often -100)
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     print("Processed Labels", labels)
#     # Decode sequences
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     correct = 0
#     total = len(decoded_labels)
#     print('total', total, "DECODED PREDICTIONS",decoded_preds, "ACTUAL LABELS", decoded_labels)
#     for pred, label in zip(decoded_preds, decoded_labels):
#         pred_functions = extract_top_function_names(pred)
#         label_functions = extract_top_function_names(label)
#         # print('pred_functions', pred_functions, 'label_functions', label_functions, set(pred_functions), set(label_functions))
#         if set(pred_functions) == set(label_functions):
#             correct += 1

#     return {"accuracy": correct / total if total > 0 else 0}

def extract_top_function_names(text: str) -> list:
    """Simplified extractor for standardized format"""
    # print('extract_top_function_names', text)
    # Match "name": "function_name" patterns
    functions = re.findall(r'"name"\s*:\s*"([^"]+)"', text)
    return list(set(functions))


def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    
    # Replace -100 in labels with pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=False)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # print('total', "DECODED PREDICTIONS",decoded_preds, "ACTUAL LABELS", decoded_labels)
    # Extract assistant's response from predictions (split after [/INST])
    assistant_preds = []
    for pred in decoded_preds:
        # Split on the last occurrence of [/INST]
        if "[/INST]" in pred:
            assistant_part = pred.split("[/INST]")[-1].strip()
        else:
            assistant_part = pred  # Fallback if delimiter missing
        assistant_preds.append(assistant_part)
    
    # Calculate accuracy based on function names
    correct = 0
    total = len(decoded_labels)
    for pred, label in zip(assistant_preds, decoded_labels):
        pred_funcs = extract_top_function_names(pred)
        label_funcs = extract_top_function_names(label)
        if set(pred_funcs) == set(label_funcs):
            correct += 1
        print('pred_funcs', pred_funcs, 'label_funcs', label_funcs, correct, set(pred_funcs), set(label_funcs))
    print('accuracy', 'correct', correct, 'total', total)
    return {"accuracy": correct / total if total > 0 else 0}
# --- TRAINER ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    compute_metrics=compute_metrics,
)

# --- TRAINING ---
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
