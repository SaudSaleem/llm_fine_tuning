import os
import wandb
from dotenv import load_dotenv
from huggingface_hub import login
from autotrain.params import LLMTrainingParams
from autotrain.project import AutoTrainProject

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
HF_USERNAME = os.getenv('HF_USERNAME')
WANDB_TOKEN = os.getenv('WANDB_TOKEN')
# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATA_PATH = "data/subset_dataset"

login(token=HF_TOKEN)
wandb.login(key=WANDB_TOKEN)
wandb.init(project="bitagent-finetune-mistral-autotrain", mode="online")

# EXTRACT 40K CHUNKS
from datasets import load_dataset, DatasetDict, load_from_disk
dataset = load_dataset("saudsaleem/mistral-7b-instruct-templated-dataset")
# Randomly sample 40k rows
random_indices = dataset["train"].shuffle(seed=42).select(range(40000))
# Split into train (80%) and test (20%) sets
train_test_split = random_indices.train_test_split(test_size=0.2, seed=42)
subset_dataset = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})
subset_dataset.save_to_disk(DATA_PATH)

dataset = load_from_disk(DATA_PATH)
print(dataset)
print(dataset["train"][0])  # Try accessing a sample
# END CODE

training_params = LLMTrainingParams(
    model=MODEL_NAME,
    data_path=DATA_PATH,
    text_column="text",
    train_split="train",
    trainer="sft",
    epochs=6,
    lr=2e-5,
    batch_size=4,
    gradient_accumulation=8,
    optimizer="adamw_torch",
    scheduler="cosine",
    peft=True,
    merge_adapter=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules="all-linear",
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    logging_steps=10,
    save_total_limit=3,
    weight_decay=0.01,
    seed=42,
    log="wandb",
    push_to_hub=True,
    username=HF_USERNAME,
    token=HF_TOKEN,
    project_name="bitagent-finetune-mistral-autotrain",
    mixed_precision="bf16" 
)

# Initialize and run trainer
backend = "local"
project = AutoTrainProject(params=training_params, backend=backend, process=True)
project.create()
