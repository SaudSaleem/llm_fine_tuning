import pandas as pd
import json
import re
import ast

# Load the CSV file
df = pd.read_csv('data-preprocessing/bitagent.data/samples/glaive_sample.csv')

types_dict = {
    'str': "string",
    'int': "integer",
    'float': "number",
    'bool': "boolean",
    'list': "array",
    'dict': "object",
}
def modify_tools(tools):
  tool = tools[0]
  # print('tool', tool)
  obj = {}
  obj['type'] = "function"
  obj['function'] = {}
  obj['function']['name'] = tool['name']
  obj['function']['description'] = tool['parameters']
  # obj['function']['parameters'] = {}
  # obj['function']['parameters']['type'] = "object"
  # obj['function']['parameters']['properties'] = {}
  # obj['function']['parameters']['required'] = []
  # for key, value in tool['parameters'].items():
  #   print('hello key', key, 'value', value)
  #   obj['function']['parameters']['properties'][key] = {}
  #   obj['function']['parameters']['properties'][key]['type'] = types_dict.get(value['type'], value['type'])
  #   obj['function']['parameters']['properties'][key]['description'] = value['description']
  #   if value['required'] == True:
  #     obj['function']['parameters']['required'].append(key)
  obj['function']['strict'] = True
  return [obj]

def extract_tools(system_str):
    try:
        start = system_str.find('{')
        end = system_str.rfind('}') + 1
        json_str = system_str[start:end]
        tool = json.loads(json_str)
        tool = modify_tools([tool])
        return json.dumps(tool)
    except json.JSONDecodeError as e:
        print(f"Error extracting tools: {e}")
        return []


# Updated regular expressions with re.DOTALL flag
user_pattern = r'USER:(.*?)(?=\s*(?:ASSISTANT:|<\|endoftext\>|$))'
assistant_pattern = r"ASSISTANT: (.*?)(?= <\|endoftext\|>|$)"

# Extracting groups with flags to handle multiline messages
df["user_groups"] = df["chat"].apply(lambda x: re.findall(user_pattern, x, flags=re.DOTALL))
df["assistant_groups"] = df["chat"].apply(lambda x: re.findall(assistant_pattern, x, flags=re.DOTALL))
def process_user_groups(user_groups_list):
    merged_content = ' '.join(user_groups_list)
    return {"role": "user", "content": merged_content}

def process_assistant_groups(assistant_groups_list):
    tool_call = None
    for msg in assistant_groups_list:
        if msg.startswith('<functioncall>'):
            json_str = msg.split('<functioncall>', 1)[1].strip()
            try:
                # Parse the outer JSON structure
                print('saud salem', json_str)
                func_data = json.loads(json_str)
                func_name = func_data.get('name', '')
                arguments = func_data.get('arguments', {})

                # Handle cases where arguments is a string or dict
                if isinstance(arguments, str):
                    # Replace single quotes and load as JSON
                    arguments = arguments.replace("'", '"')
                    args_dict = json.loads(arguments)
                else:
                    # Use the dictionary directly
                    args_dict = arguments

                # Format keyword arguments
                args_kwargs = ', '.join(f"{k}={v}" for k, v in args_dict.items())
                tool_call = f"{func_name}({args_kwargs})"
                break  # Process only the first function call found
            except (SyntaxError, json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"Error processing function call: {e}", msg)
                continue
    return {"role": "assistant", "content": tool_call}

# Apply processing to create the conversation column
df['conversation'] = df.apply(lambda row: json.dumps([
    process_user_groups(row['user_groups']),
    process_assistant_groups(row['assistant_groups'])
]), axis=1)
# Filter rows where ANY assistant message has empty content
df = df[~df['conversation'].apply(
    lambda conv: any(
        (msg.get('role') == 'assistant') and
        (msg.get('content') is None or msg.get('content') == '')
          # This prints the message being checked
        for msg in json.loads(conv)
    )
)]
df['tools'] = df['system'].apply(extract_tools)
df = df[["tools", "conversation"]]
# Display the result
df.to_csv('data-preprocessing/bitagent.data/samples/glaive_modified.csv', index=False)