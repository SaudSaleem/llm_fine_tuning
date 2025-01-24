# LOAD THE BASE MODEL
import os
import torch
import shutil
import optuna
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from huggingface_hub import HfApi, HfFolder
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# Clear cache by deleting the model folder from cache
cache_dir = os.path.expanduser("~/.cache/huggingface")
shutil.rmtree(cache_dir, ignore_errors=True)
model_save_path = "fine-tuned-mistral"
if os.path.exists(model_save_path):
    shutil.rmtree(model_save_path)
    print(f"Model saved at {model_save_path} deleted.")


print('Cude is available: ', torch.cuda.is_available())  # Should return True if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("Cuda version: ", torch.version.cuda)  # Prints the CUDA version that PyTorch is compiled with

# Load environment variables from the .env file
load_dotenv()

# Load quantized model and tokenizer
model_name = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set the padding token if it's not already defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",       # Automatically map layers to available GPUs
    torch_dtype="auto",
    trust_remote_code=True   
)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# LOAD AND PREPARE THE DATASET

df = pd.read_csv("bitAgent.csv")
# Ensure dataset is in the correct format (message-based JSON)
df = df.rename(columns={"input": "user", "output": "assistant"})
dataset = Dataset.from_pandas(df)

# Split into training and validation sets
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
val_dataset = train_test_split["test"]

print("train dataset",train_dataset[0])
# Tokenize dataset
def preprocess(example):
    # Extracting the 'content' key from the 'user' column for the prompt
    prompt = example['user']
    
    # Tokenizing the prompt
    tokens = tokenizer(
        prompt,  # Use the 'content' key as the prompt
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    
    # If you want to treat the same tokens as labels, you can do this:
    tokens["labels"] = tokens["input_ids"].copy()  # Make labels same as input_ids
    
    return tokens


train_dataset = train_dataset.map(preprocess, batched=True)
val_dataset = val_dataset.map(preprocess, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
print('MODEL SAUD', model)

print('printing self aten')
for name, param in model.named_parameters():
    if 'self_attn.q_proj' in name or 'self_attn.k_proj' in name or 'self_attn.v_proj' in name:
        print(f"LoRA Layer {name}: requires_grad={param.requires_grad}")

print('LORA RELATED PRINTS end')

for name, module in model.named_modules():
    if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
        print(f"Found projection layer: {name}")

for name, param in model.named_parameters():
    print(f"Layer: {name}, requires_grad: {param.requires_grad}")
    if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
        print(f"LoRA Layer {name}: requires_grad={param.requires_grad}")




# Configure LoRA
lora_config = LoraConfig(
    target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],  # Full path to the layers
    r=8,  # Rank of the low-rank approximation
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1  # Dropout rate
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# function for hyperparameter tuning
def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    per_device_train_batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    gradient_accumulation_steps = trial.suggest_categorical("grad_accum", [1, 2, 4])

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        logging_dir="./logs",
        fp16=True,  # Mixed precision
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="none",  # Log with Weights & Biases
        # run_name="mistral_run_name",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results["eval_loss"]

# Run Optuna Study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# Get the best hyperparameters
best_hyperparameters = study.best_params
print(best_hyperparameters)

# Fine-Tuning
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=best_hyperparameters["learning_rate"],
    per_device_train_batch_size=best_hyperparameters["batch_size"],
    gradient_accumulation_steps=best_hyperparameters["grad_accum"],
    num_train_epochs=3,
    logging_dir="./logs",
    fp16=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    # run_name="mistral_run_name",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()


# Save model and tokenizer locally
model.save_pretrained("fine-tuned-mistral")
tokenizer.save_pretrained("fine-tuned-mistral")



# Authenticate with your Hugging Face token
# Retrieve the token
hf_token = os.getenv("HF_TOKEN")
HfFolder.save_token(hf_token)
repo_id = "saudsaleem/fine-tuned-mistral"

# Create the repository as public
api = HfApi()
api.create_repo(repo_id=repo_id, private=False)

# Push model and tokenizer to the public repository
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)

# Inference Script

# Load the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("saudsaleem/fine-tuned-mistral")
model = AutoModelForCausalLM.from_pretrained("saudsaleem/fine-tuned-mistral")

# Generate response
def generate_response(input_text):
    inputs = tokenizer(f"User: {input_text}\nAssistant:", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generate_response("What is the weather in NYC?"))