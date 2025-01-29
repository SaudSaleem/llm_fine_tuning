import torch
import os
import json
import re
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
    AutoConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    GenerationConfig
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
DATASET_PATH = "bitAgent.csv"
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
    system_prompt = """Respond ONLY with valid JSON containing tool calls. Example:
    User: play Happy Song
    Assistant: [{"play_song": {"query": "Happy Song"}}]"""
    
    return {
        "user": f"[INST] {system_prompt}\n\nInput: {row['input']} [/INST]",
        "assistant": f"{row['output']}</s>"
    }

# Create a list of formatted examples
formatted_data = df.apply(format_training_example, axis=1).tolist()

# Convert to DataFrame first
formatted_df = pd.DataFrame(formatted_data)

# Then create the Dataset
dataset = Dataset.from_pandas(df)
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
    print('preprocess_function', examples,"1234567", examples['user'])
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
    num_train_epochs=20,
    learning_rate=2e-3,
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

def compute_metrics(eval_pred):
    preds = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    
    # Decode predictions
    decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)
    valid = sum(validate_json_output(d) for d in decoded) / len(decoded)
    print('accuracy SAUD SALEM', valid)
    return {"json_accuracy": valid}

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
    system_msg = """Respond ONLY with valid JSON tool calls. Example:
    Input: play Song Name
    Output: [{"play_song": {"query": "Song Name"}}]"""
    
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