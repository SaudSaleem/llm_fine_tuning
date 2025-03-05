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

# import os


login(token=HF_TOKEN)
wandb.login(key=WANDB_TOKEN)
wandb.init(project="bitagent-finetune-qwen-autotrain", mode="online")
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
DATA_PATH = "data/csv/"

login(token=HF_TOKEN)
wandb.login(key=WANDB_TOKEN)
wandb.init(project="bitagent-finetune-qwen2-autotrain", mode="online")

# Add validation monitoring
training_params = LLMTrainingParams(
    model=MODEL_NAME,
    data_path=DATA_PATH,
    text_column="text",
    train_split="train",
    valid_split="test",  # Add validation split if available
    trainer="sft",
    epochs=10,  # Increased to 10 epochs as requested
    lr=1e-5,  # Slightly lower learning rate for longer training
    batch_size=20,  # Smaller batch size
    gradient_accumulation=8,  # Increased for effective batch size of 64
    optimizer="adamw_torch",
    scheduler="cosine",
    peft=True,
    merge_adapter=True,
    lora_r=32,  # Increased for more capacity
    lora_alpha=64,  # 2x lora_r
    lora_dropout=0.1,
    # More comprehensive target modules for deeper training
    target_modules="q_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    logging_steps=5,
    eval_steps=100,  # Add evaluation steps to monitor validation performance
    save_total_limit=3,
    weight_decay=0.001,
    seed=42,
    log="wandb",
    push_to_hub=True,
    username=HF_USERNAME,
    token=HF_TOKEN,
    project_name="bitagent-finetune-qwen2-autotrain",
    mixed_precision="bf16",
    save_strategy="steps",
    save_steps=200,  # Save checkpoints to analyze training progress
)
# Initialize and run trainer
backend = "local"
project = AutoTrainProject(params=training_params, backend=backend, process=True)
project.create()