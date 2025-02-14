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
DATA_PATH = "data/csv/"

login(token=HF_TOKEN)
wandb.login(key=WANDB_TOKEN)
wandb.init(project="bitagent-finetune-mistral-autotrain", mode="online")


training_params = LLMTrainingParams(
    model=MODEL_NAME,
    data_path=DATA_PATH,
    text_column="text",
    train_split="train",
    trainer="sft",
    epochs=6,
    lr=2e-5,
    batch_size=20,
    gradient_accumulation=2,
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
