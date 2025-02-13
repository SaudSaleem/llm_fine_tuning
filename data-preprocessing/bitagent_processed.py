import json
import pandas as pd
from ast import literal_eval

# Load the CSV file
df = pd.read_csv('data-preprocessing/bitagent.data/samples/bitagent_sample.csv')

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
  obj['function']['description'] = tool['description']
  obj['function']['parameters'] = {}
  obj['function']['parameters']['type'] = "object"
  obj['function']['parameters']['properties'] = {}
  obj['function']['parameters']['required'] = []
  for key, value in tool['arguments'].items():
    obj['function']['parameters']['properties'][key] = {}
    obj['function']['parameters']['properties'][key]['type'] = types_dict.get(value['type'], value['type'])
    obj['function']['parameters']['properties'][key]['description'] = value['description']
    if value['required'] == True:
      obj['function']['parameters']['required'].append(key)
  # obj['function']['strict'] = True
  return [obj]
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
    df.at[index, 'tools'] = json.dumps(modify_tools(tools))
    # Convert the conversation back to a JSON string
    df.at[index, 'conversation'] = json.dumps(conversation)

# Save the modified DataFrame
df.to_csv('data-preprocessing/bitagent.data/samples/bitagent_modified.csv', index=False)