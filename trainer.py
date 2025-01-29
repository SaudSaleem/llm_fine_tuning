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
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    GenerationConfig,
    EvalPrediction
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
import transformers

# --- CONFIGURATION ---
MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
DATASET_PATH = "bitAgent1.csv"
OUTPUT_DIR = "/home/user/saud/models/fine-tuned-mistral-bitagent-latest"
HF_TOKEN = os.getenv("HF_TOKEN")

# --- JSON RESPONSE STOPPING CRITERIA ---
class JsonStopCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stack = []
        self.brace_count = 0
        
    def __call__(self, input_ids, scores, **kwargs):
        last_token = input_ids[0][-1]
        decoded = self.tokenizer.decode([last_token])
        
        if decoded == '{':
            self.brace_count += 1
            self.stack.append('}')
        elif decoded == '[':
            self.stack.append(']')
        elif decoded in ['}', ']']:
            if self.stack and self.stack[-1] == decoded:
                self.stack.pop()
                if decoded == '}': 
                    self.brace_count -= 1
                    
        # Stop when JSON structure is complete
        if self.brace_count == 0 and len(self.stack) == 0:
            return True
        return False

# --- LOGIN TO HUGGINGFACE ---
login(token=HF_TOKEN)

# --- LOAD AND PREPROCESS DATASET ---
df = pd.read_csv(DATASET_PATH)

# Add system prompt to training data
def format_training_example(row):
    system_prompt = """Respond ONLY with function call. Example:
    User: What is the distance from Los Angeles to New York
    Assistant: calculate_distance(destination="New York", origin="Los Angeles")"""
    
    return {
        "user": f"[INST] {system_prompt}\n\nInput: {row['input']} [/INST]",
        "assistant": f"{row['output']}</s>"
    }

# Create a list of formatted examples
formatted_df = df.apply(format_training_example, axis=1)

# Convert the formatted dataframe into a new DataFrame with 'user' and 'assistant' columns
formatted_df = pd.DataFrame(formatted_df.tolist())

# Convert to DataFrame first
# formatted_df = pd.DataFrame(formatted_df)

# Then create the Dataset
dataset = Dataset.from_pandas(formatted_df)
# print('dataset', dataset, dataset[0])
train_test_split = dataset.train_test_split(test_size=0.15, seed=42)


# --- MODEL SETUP ---
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # quantization_config=bnb_config,
    device_map="auto",
    # attn_implementation="flash_attention_2"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

# --- DATA PROCESSING ---
def preprocess_function(examples):
    print('saleem', examples)
    tokenized = tokenizer(
        examples["user"],
        text_target=examples["assistant"],
        max_length=768,
        truncation=True,
        padding="max_length",
    )
    
    # Mask user input in labels
    user_tokens = tokenizer(examples["user"], add_special_tokens=False)["input_ids"]
    labels = [-100]*len(user_tokens) + tokenized["labels"][len(user_tokens):]
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

tokenized_ds = train_test_split.map(
    preprocess_function,
    batched=True,
    batch_size=128
)

# --- LORA CONFIG ---
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    bias="lora_only",
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head"]
)
model = get_peft_model(model, peft_config)

# --- TRAINING ARGS ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=1e-3,
    optim="paged_adamw_32bit",
    logging_steps=10,
    eval_strategy="epoch",
    eval_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    bf16=True,
    max_grad_norm=0.5,
    report_to="none",
    gradient_checkpointing=True,
    # generation_config={
    #     "max_new_tokens": 256,
    #     "temperature": 0.7,
    #     "top_p": 0.9,
    #     "repetition_penalty": 1.15
    # }
)

# --- VALIDATION ---
def validate_json_output(text):
    try:
        json_str = re.search(r'\[.*\]', text, re.DOTALL).group()
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert all(isinstance(item, dict) for item in parsed)
        return True
    except:
        return False

# def extract_function_name(prediction):
#     """Extract function name from model's prediction."""
#     try:
#         print(type(prediction), 'before')

#         # Check if the prediction is an ndarray (embedding)
#         if isinstance(prediction, np.ndarray):
#             prediction = prediction.tolist()  # Convert ndarray to list
#         print(type(prediction), 'after')

#         # If prediction is tokenized (list of token IDs), decode it
#         if isinstance(prediction, list):
#             # Assuming the list contains token IDs, decode them
#             decoded_prediction = tokenizer.decode(prediction, skip_special_tokens=True)
#             print('decoded_prediction', decoded_prediction)
#             return decoded_prediction
#         # If prediction is already a string (decoded text)
#         elif isinstance(prediction, str):
#             return prediction
#     except Exception as e:
#         print(f"Error: {e}")
#         return ""
#     return ""

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     metric = load("accuracy")
#     f1_metric = load("f1")
#     precision_metric = load("precision")
#     recall_metric = load("recall")
    
#     predictions = [extract_function_name(pred) for pred in predictions]
#     labels = [extract_function_name(label) for label in labels]
    
#     accuracy = metric.compute(predictions=predictions, references=labels)
#     f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
#     precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
#     recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")
#     print('accuracy', accuracy['accuracy'])
#     return {
#         "accuracy": accuracy["accuracy"],
#         "f1": f1["f1"],
#         "precision": precision["precision"],
#         "recall": recall["recall"]
#     }



def compute_metrics(eval_pred: EvalPrediction):
    # Get predictions and labels
    predictions, labels = eval_pred
    
    # Convert logits to token IDs (assuming classification head)
    predictions = np.argmax(predictions, axis=-1)
    
    # Remove ignored indices (often -100)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode sequences
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    correct = 0
    total = len(decoded_labels)
    print('total', total, decoded_preds, decoded_labels)
    for pred, label in zip(decoded_preds, decoded_labels):
        pred_functions = extract_top_function_names(pred)
        label_functions = extract_top_function_names(label)
        print('pred_functions', pred_functions, 'label_functions', label_functions, set(pred_functions), set(label_functions))
        if set(pred_functions) == set(label_functions):
            correct += 1

    return {"accuracy": correct / total if total > 0 else 0}

def extract_top_function_names(text: str) -> list:
    """Simplified extractor for standardized format"""
    print('extract_top_function_names', text)
    # Match "name": "function_name" patterns
    functions = re.findall(r'"name"\s*:\s*"([^"]+)"', text)
    return list(set(functions))



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

# --- INFERENCE ---
def generate_tool_call(query, device="cuda"):
    system_msg = """Respond ONLY with function call. Example:
    User: What is the distance from Los Angeles to New York
    Assistant: calculate_distance(destination="New York", origin="Los Angeles")"""
    
    
    prompt = f"[INST] {system_msg}\n\nInput: {query} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    stopping_criteria = StoppingCriteriaList([JsonStopCriteria(tokenizer)])
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15,
        stopping_criteria=stopping_criteria,
        do_sample=False
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    json_str = re.search(r'\[.*\]', response, re.DOTALL).group()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}

# Example usage
# if __name__ == "__main__":
    # tool_call = generate_tool_call("play Johnny Johnny Yes papa")
    # print(json.dumps(tool_call, indent=2))