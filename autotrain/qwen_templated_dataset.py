import os
import json
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict

# Load environment variables from .env file
load_dotenv()

# Configuration
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # Updated to Qwen2-7B-Instruct
MAX_SEQ_LENGTH = 1024
DATASET_PATH = "../bitAgent.csv"
OUTPUT_DIR = "autotrain-output"

# Login to Hugging Face
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
            
            # Format conversation with tools in system message
            formatted_conversation = []
            
            # Add system message with tools
            system_content = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.

You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke:"""
            
            system_content += f"\n{json.dumps(tools, indent=2)}"
            
            # Add system message
            formatted_conversation.append({"role": "system", "content": system_content})
            
            # Add the rest of the conversation
            for message in conversation:
                formatted_conversation.append(message)
            
            # Apply chat template
            full_tokenized = tokenizer.apply_chat_template(
                formatted_conversation,
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

# Filter empty strings
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
dataset_dict.push_to_hub(f"{os.getenv('HF_USERNAME')}/Meta-Llama-3-8B-Instruct-templated-dataset")

print("Dataset pushed successfully!")