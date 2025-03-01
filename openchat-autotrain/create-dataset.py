from datasets import Dataset, load_dataset
import random
import json
import pandas as pd

def system_prompt(tools):
    prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
    If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
    You should only return the function call in tools call sections.

    If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1="params_string_value1", params_name2=params_value2...), func_name2(params)]
    Notice that any values that are strings must be put in quotes like this: "params_string_value1"
    You SHOULD NOT include any other text in the response.
    Here is a list of functions in JSON format that you can invoke.\n{functions}\n
    """
    return prompt.format(functions=tools)

# Load dataset from Hugging Face
dataset = load_dataset("BitAgent/tool_shuffle_small", split="train")

def process_example(example, all_tools):
    # Get random number of additional tools (1-5) + original tool
    num_extra = random.randint(1, 5)
    extra_indices = random.sample(range(len(all_tools)), num_extra)
    combined_tools = [example["tools"]] + [all_tools[i] for i in extra_indices]
    
    # Create system message with all tools
    system_msg = {
        "role": "system",
        "content": system_prompt(json.dumps(combined_tools, indent=2))
    }
    
    # Prepend system message to existing conversation
    conversation = json.loads(example["conversation"])  # Parse JSON string to list
    new_conversation = [system_msg] + conversation
    
    return {"text": json.dumps(new_conversation)}

# Extract all tools from dataset for sampling
all_tools = [ex["tools"] for ex in dataset]

# Process dataset with augmented tools
processed_dataset = dataset.map(
    process_example,
    fn_kwargs={"all_tools": all_tools},
    remove_columns=["conversation", "tools"]
)

# Convert to pandas and save as CSV
df = processed_dataset.to_pandas()
df.to_csv("processed_dataset.csv", index=False, escapechar='\\')
