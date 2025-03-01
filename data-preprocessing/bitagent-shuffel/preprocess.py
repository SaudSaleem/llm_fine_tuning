import json
import pandas as pd
from ast import literal_eval
import datasets
import random
# Load dataset from Hugging Face
# dataset = datasets.load_dataset("BitAgent/tool_shuffle_small", split="train")
# df = pd.DataFrame(dataset)
# df.to_csv('bitagent_shuffel.csv', index=False)

df = pd.read_csv('bitagent_shuffel.csv')

def format_arguments(arguments):
    """Formats function arguments into a string suitable for a function call."""
    formatted_args = []
    # print('arguments', arguments)
    for key, value in arguments.items():
        if isinstance(value, str):
            # Escape single quotes and wrap in single quotes
            formatted_value = value.replace("'", r"\'")
        elif isinstance(value, bool):
            formatted_value = 'True' if value else 'False'
        elif isinstance(value, (int, float)):
            formatted_value = str(value)
        elif value is None:
            formatted_value = 'None'
        else:
            # Use JSON serialization for non-string types (lists, dicts, etc.)
            formatted_value = json.dumps(value)
        formatted_args.append(f"{key}='{formatted_value}'")
    return ", ".join(formatted_args)


# Process each row in the DataFrame
for index, row in df.iterrows():
    # Parse the conversation string into a list of dictionaries
    try:
        conversation = json.loads(row['conversation'])
        tools = json.loads(row['tools'])
    except json.JSONDecodeError:
        conversation = literal_eval(row['conversation'])

    # Track tool calls to modify subsequent assistant messages
    for i in range(len(conversation)):
        try:
          entry = conversation[i]
          if entry['role'] == 'tool call':
              content = entry['content']
              name = content.get('name', '')
              arguments = content.get('arguments', {})
              # Update the next assistant entry if present
              if i + 1 < len(conversation) and conversation[i+1]['role'] == 'assistant':
                  assistant_entry = conversation[i+1]
                  args_str = format_arguments(arguments)
                  function_call = f"{name}({args_str})" if args_str else f"{name}()"
                  assistant_entry['content'] = function_call
        except Exception as e:
          print('error', e)
        

    # Remove all entries where role is 'tool call'
    conversation = [entry for entry in conversation if entry['role'] != 'tool call']

    # modify tools structure according to openAI function calling schema
    df.at[index, 'tools'] = json.dumps(tools)
    # Convert the conversation back to a JSON string
    df.at[index, 'conversation'] = json.dumps(conversation)


# df.to_csv('bitagent_shuffel_processed.csv', index=False)
# # ADD EXTRAB TOOLS IN TOOLS COLUMNS
# _df = pd.read_csv('bitagent_shuffel_processed.csv')
# Collect all tools from all rows
all_tools = []
for tools_str in df['tools']:
    # print('tools_str', tools_str)
    tools = json.loads(tools_str)
    all_tools.extend(tools)

errored_rows = 0
# Update tools column with JSON-safe formatting
def update_tools(tools_str):
  try:
      global errored_rows
      selected_tool = random.choice(all_tools)
      tools_list = json.loads(tools_str)
      # Determine the number of tools to add (2-6)
      num_tools_to_add = random.randint(2, 6)
      for _ in range(num_tools_to_add):
        selected_tool = random.choice(all_tools)
        # Randomly choose to append or prepend each tool
        if random.choice([True, False]):
            tools_list.append(selected_tool)
        else:
            tools_list.insert(0, selected_tool)
      return json.dumps(tools_list)
  except Exception as e:
        errored_rows += 1
        print(f"Unexpected error: {e}", tools_str)
        return tools_str

df['tools'] = df['tools'].apply(update_tools)

# Save the modified DataFrame
df.to_csv('bitagent_shuffel_processed.csv', index=False)