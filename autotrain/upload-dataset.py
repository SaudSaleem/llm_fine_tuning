import os
import json
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict

# Load environment variables from .env file
load_dotenv()
# Configuration
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_SEQ_LENGTH = 1024
DATASET_PATH = "../bitAgent.csv"
OUTPUT_DIR = "autotrain-output"

login(token=os.getenv("HF_TOKEN"))

# Load and verify dataset
dataset = load_dataset('csv', data_files=DATASET_PATH)['train']

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    batch = {
        "text": []
    }
    for conv_str, tools_str in zip(examples['conversation'], examples['tools']):
        try:
            # Parse JSON columns
            conversation = json.loads(conv_str)
            tools = json.loads(tools_str)
            full_tokenized = tokenizer.apply_chat_template(
            conversation,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False
            )
            batch["text"].append(full_tokenized)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping invalid row: {e}")
            continue
    return batch
# Process dataset in batches
formatted_ds = dataset.map(tokenize_function, batched=True, batch_size=2, remove_columns=dataset.column_names)
print('formatted_ds', formatted_ds)  

# Filter empty strings (no need to convert to HF_dataset again since formatted_ds is already a Dataset)
filtered_ds = formatted_ds.filter(lambda x: len(x['text'].strip()) > 0)

# Split dataset into train (80%) and test (20%)
train_test_split = filtered_ds.train_test_split(test_size=0.2)

# Wrap in DatasetDict for Hugging Face Hub compatibility
dataset_dict = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})
print('dataset_dict', dataset_dict)

# Push dataset to Hugging Face Hub
dataset_dict.push_to_hub(f"{os.getenv('HF_USERNAME')}/mistral-7b-instruct-templated-dataset")

print("Dataset pushed successfully!")
