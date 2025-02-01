import os
from sklearn.metrics import precision_recall_fscore_support
import re
from typing import List, Dict, Any
from datasets import Dataset
import pandas as pd
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EvalPrediction,
    AutoConfig
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
OUTPUT_DIR = "/home/user/saud/models/fine-tuned-mistral-bitagent-latest"
HF_TOKEN = os.getenv("HF_TOKEN")
# Adjust these weights based on importance
WEIGHT_FUNCTION = 0.7
WEIGHT_ARGUMENTS = 0.3

# --- LOGIN TO HUGGINGFACE ---
# login(token=HF_TOKEN)
# wandb.login()
# os.environ["WANDB_PROJECT"] = "mistral-finetune"
# --- LOAD AND PREPROCESS DATASET ---
df = pd.read_csv(DATASET_PATH)

# Add system prompt to training data
def format_training_example(row):
    return {
        "messages": [
            {"role": "user", "content": row['input']},
            {"role": "assistant", "content": row['output']}
        ]
    }

# Create a list of formatted examples
formatted_df = df.apply(format_training_example, axis=1)
# Convert the formatted dataframe into a new DataFrame with 'user' and 'assistant' columns
formatted_df = pd.DataFrame(formatted_df.tolist())
formatted_df = formatted_df.iloc[:1000] 
# Then create the Dataset
dataset = Dataset.from_pandas(formatted_df)
train_test_split = dataset.train_test_split(test_size=0.15, seed=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto",)
model = prepare_model_for_kbit_training(model)

# --- DATA TOKENIZATION ---
def tokenize_function(example):
    messages = example['messages']
    # Tokenize the entire conversation
    tokenized = tokenizer.apply_chat_template(messages, truncation=True)
    # Tokenize user messages to find where the assistant's response starts
    user_messages = [messages[0]]
    user_prompt = tokenizer.apply_chat_template(user_messages, truncation=True, add_generation_prompt=True)
    user_length = len(user_prompt)
    # Create labels: mask user part, keep assistant part
    labels = [-100] * user_length + tokenized[user_length:]
    return {
        "input_ids": tokenized,
        "attention_mask": [1] * len(tokenized),
        "labels": labels
    }


tokenized_ds = train_test_split.map(
    tokenize_function,
    batched=False,
    # batch_size=64
)
tokenn = tokenized_ds["train"][0]
# Print the raw tokenized dictionary
print("test 123 tokenizer:", tokenn)

# Decode input tokens
decoded_text = tokenizer.decode(tokenn["input_ids"])
print("Decoded Input:", decoded_text)

# Decode label tokens
label_text = tokenizer.decode(tokenn["labels"], skip_special_tokens=True)
print("Decoded Labels:", label_text)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
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
    """
    Computes weighted function accuracy and F1-score for argument matching.
    """
    predictions, references = eval_pred
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
    learning_rate=2e-5,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    load_best_model_at_end=True,
    gradient_checkpointing=True,
    metric_for_best_model="f1_score_args", 
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
    compute_metrics=compute_metrics,
)

# --- TRAINING ---
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
# Load the configuration from the fine-tuned model and save it to the same directory
config = AutoConfig.from_pretrained(OUTPUT_DIR)
config.save_pretrained(OUTPUT_DIR)
