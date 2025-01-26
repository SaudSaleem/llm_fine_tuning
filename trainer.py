import torch
import os
from datasets import Dataset
import pandas as pd
from huggingface_hub import notebook_login, login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import transformers
from datetime import datetime
from peft import PeftModel



# LOAD AND PREPARE THE DATASET

df = pd.read_csv("bitAgent.csv")
# Ensure dataset is in the correct format (message-based JSON)
df = df.rename(columns={"input": "user", "output": "assistant"})
dataset = Dataset.from_pandas(df)

# Split into training and validation sets
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
val_dataset = train_test_split["test"]

print("train dataset", train_dataset[0])

hf_token = os.getenv("HF_TOKEN")
print('hf_token', hf_token)
login(token=hf_token)
# notebook_login()


base_model_id = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(base_model_id, 
                                            #  quantization_config=bnb_config, 
                                             device_map="auto")

# Step 4: Tokenization
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

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


tokenized_train_dataset = train_dataset.map(preprocess)
tokenized_val_dataset = val_dataset.map(preprocess)

# Step 6: Tokenize with padding and truncation
max_length = 512

# Step 8: Set Up LoRA
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


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
# print_trainable_parameters(model)

config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        # "lm_head",
    ],
    bias="none",
    lora_dropout=0.15,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze specific layers for fine-tuning (e.g., LoRA adapter)
for param in model.lora.parameters():
    param.requires_grad = True
print_trainable_parameters(model)

# Step 9: Set up Trainer
if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

project = f"{base_model_id} + bitagent-finetune"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=500,
        learning_rate=2.5e-5,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=25,
        evaluation_strategy="steps",
        eval_steps=25,
        do_eval=True,
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        load_best_model_at_end=True,
        save_total_limit= 1,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False
trainer.train()
# Save model and tokenizer locally
output_dir = "/home/user/saud/models/fine-tuned-mistral-bitagent-latest"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Load the configuration from the fine-tuned model and save it to the same directory
# config = AutoConfig.from_pretrained(output_dir)
# config.save_pretrained(output_dir)
# Verification
print(f"Fine-tuned model saved to {output_dir} with the following files:")
print(os.listdir(output_dir))

# Step 10: Evaluation After Fine-Tuning
# base_model_id = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_id,
#     # quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True,
# )

# eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

# eval_prompt = " The following is a note by Eevee the Dog, which doesn't share anything too personal: # "
# model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

# ft_model = PeftModel.from_pretrained(base_model, "mistral-journal-finetune/checkpoint-300")

# ft_model.eval()
# with torch.no_grad():
#     print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15)[0], skip_special_tokens=True))