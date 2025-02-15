import os
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
import re
import numpy as np
from typing import List, Dict, Any
from datasets import Dataset, load_dataset
import pandas as pd
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EvalPrediction,
    AutoConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)


# --- CONFIGURATION ---
MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
DATASET_PATH = "bitAgent.csv"
OUTPUT_DIR_LOGS = "/home/user/saud/models/logs"
OUTPUT_DIR = Path.home() / "models" / "fine-tuned-mistral-bitagent"
HF_TOKEN = os.getenv("HF_TOKEN")
# Adjust these weights based on importance
WEIGHT_FUNCTION = 0.7
WEIGHT_ARGUMENTS = 0.3

# --- LOGIN TO HUGGINGFACE ---
# login(token=HF_TOKEN)
# wandb.login()
# os.environ["WANDB_PROJECT"] = "mistral-finetune"
# --- LOAD AND PREPROCESS DATASET ---
# df = pd.read_csv(DATASET_PATH)
dataset = load_dataset('csv', data_files=DATASET_PATH)
dataset = dataset['train'].select(range(100000))
# Display a sample
print("said", dataset)
# Add system prompt to training data
def format_training_example(example):
    instruction = example['input']
    response = example['output']
    formatted_text = f"<s>[INST] {instruction} [/INST] {response}</s>"
    return {'text': formatted_text}

formatted_dataset = dataset.map(format_training_example)
# formatted_dataset = formatted_dataset.select(range(1000))
train_test_split = formatted_dataset.train_test_split(test_size=0.15, seed=42)
print('train_test_split', train_test_split, train_test_split['train'][0])
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto",)
model = prepare_model_for_kbit_training(model)

context_length = 128
# --- DATA TOKENIZATION ---
# def tokenize_function(example):
#     messages = example['messages']
#     # Tokenize the entire conversation
#     tokenized = tokenizer.apply_chat_template(messages, truncation=True, max_length=context_length)
#     # Tokenize user messages to find where the assistant's response starts
#     user_messages = [messages[0]]
#     user_prompt = tokenizer.apply_chat_template(user_messages, truncation=True, max_length=context_length, add_generation_prompt=True)
#     user_length = len(user_prompt)
#     # Create labels: mask user part, keep assistant part
#     labels = [-100] * user_length + tokenized[user_length:]
#     attention_mask = [1] * len(tokenized)
#     print('user_prompt', len(tokenized), len(labels), 'attenion mask', len(attention_mask))
#     return {
#         "input_ids": tokenized,
#         "attention_mask": attention_mask,
#         "labels": labels
#     }

print("formatted_dataset",formatted_dataset[0])
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_ds = train_test_split.map(tokenize_function, batched=True)
# tokenized_ds = train_test_split.map(
#     tokenize_function,
#     batched=False,
#     # batch_size=64
# )
tokenn = tokenized_ds["train"][0]
# Print the raw tokenized dictionary
print("test 123 tokenizer:", tokenn, 'tokenized_ds', tokenized_ds)

# Decode input tokens
decoded_text = tokenizer.decode(tokenn["input_ids"])
print("Decoded Input:", decoded_text)

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
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.enable_input_require_grads()
print_trainable_parameters(model)

def extract_top_function_names(text: str) -> list:
    """Simplified extractor for standardized format"""
    print('extract_top_function_names', text)
    # Match "name": "function_name" patterns
    functions = re.findall(r'"name"\s*:\s*"([^"]+)"', text)
    return list(set(functions))

def extract_function_parts(call):
    """ Extract function name and arguments from function call string """
    print('extract_function_parts', call)
    match = re.match(r'(\w+)\((.*?)\)', call)
    if match:
        func_name = match.group(1)
        args = match.group(2).split(",") if match.group(2) else []
        args = [arg.strip().strip('"') for arg in args]  # Remove spaces and quotes
        return func_name, args
    return call, []  # Return original if format is incorrect

def compute_metrics(eval_pred):
    print('compute_metrics called')
    """
    Computes weighted function accuracy and F1-score for argument matching.
    """
    predictions, references = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    references = np.argmax(predictions, axis=-1)
    correct_func = 0
    total_funcs = len(predictions)

    all_preds, all_refs = [], []

    for pred, ref in zip(predictions, references):
        pred_func, pred_args = extract_function_parts(pred)
        ref_func, ref_args = extract_function_parts(ref)
        print('compute_metrics', pred_func, pred_args, 'label', ref_func, ref_args)
        # Check if function names match
        if pred_func == ref_func:
            correct_func += 1

        # Collect arguments for F1 calculation
        all_preds.extend(pred_args)
        all_refs.extend(ref_args)

    # Compute function accuracy
    function_accuracy = correct_func / total_funcs

    # Compute Precision, Recall, and F1-score at the argument level
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_refs, all_preds, average='micro', zero_division=1
    )

    # Compute weighted final score
    final_score = (WEIGHT_FUNCTION * function_accuracy) + (WEIGHT_ARGUMENTS * f1)
    result = {
        "function_accuracy": function_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score_args": f1,
        "final_weighted_score": final_score
    }
    print('accuracy result', result)
    return result
# --- TRAINING ARGS ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR_LOGS,
    num_train_epochs=10,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    # save_steps=30,
    save_strategy="steps",
    load_best_model_at_end=True,
    gradient_checkpointing=True,
    per_device_train_batch_size=2,  # Adjust based on available memory
    per_device_eval_batch_size=2,   # Match with train batch size
    gradient_accumulation_steps=8,  # To maintain larger effective batch size
    fp16=True,  # Enables mixed precision training
    # metric_for_best_model="f1_score_args", 
    # report_to="wandb",  # Enable wandb logging
    # run_name="mistral-finetune-run",  # Custom run name
)
# --- TRAINER ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
)

# --- TRAINING ---
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
# Load the configuration from the fine-tuned model and save it to the same directory
config = AutoConfig.from_pretrained(OUTPUT_DIR)
config.save_pretrained(OUTPUT_DIR)