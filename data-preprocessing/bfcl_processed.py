import pandas as pd
import ast
import json

# Read the original CSV
df = pd.read_csv('data-preprocessing/bitagent.data/samples/bfcl_sample.csv')

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
  obj['function']['strict'] = True
  return [obj]

def process_row(row):
    # Parse the nested question structure
    raw_question = ast.literal_eval(row['question'])[0]
    conversation = raw_question  # Extract inner list of user messages

    # Parse ground truth to get function details
    ground_truth = ast.literal_eval(row['ground_truth'])
    func_name = list(ground_truth[0].keys())[0]
    func_args = ground_truth[0][func_name]

    # Format function arguments with proper string handling
    formatted_args = []
    for arg_name, arg_value in func_args.items():
        if isinstance(arg_value, list):
            # Handle list arguments with proper string elements
            items = [f"'{item}'" if isinstance(item, str) else str(item)
                    for item in arg_value]
            formatted_args.append(f"{arg_name}=[{', '.join(items)}]")
        else:
            # Handle scalar values
            if isinstance(arg_value, str):
                formatted_args.append(f"{arg_name}='{arg_value}'")
            else:
                formatted_args.append(f"{arg_name}={arg_value}")

    # Create assistant response
    assistant_response = {
        "role": "assistant",
        "content": f"{func_name}({', '.join(formatted_args)})"
    }
    if isinstance(conversation, dict):
      conversation = [conversation]
    # Process tools specification
    tools = ast.literal_eval(row['function'])
    for tool in tools:
        if 'parameters' in tool:
            params = tool.pop('parameters')
            arguments = {}
            required_params = params.get('required', [])

            # Convert parameters to arguments format
            for param_name, param_details in params.get('properties', {}).items():
                arguments[param_name] = {
                    "type": param_details.get('type', ''),
                    "description": param_details.get('description', ''),
                    "required": param_name in required_params
                }
            tool['arguments'] = arguments
    return {
        "conversation": conversation,
        "tools": modify_tools(tools)
    }

# Process all rows
processed_data = [process_row(row) for _, row in df.iterrows()]

# Create new DataFrame with proper JSON formatting
new_df = pd.DataFrame(processed_data)
new_df['conversation'] = new_df['conversation'].apply(json.dumps)
new_df['tools'] = new_df['tools'].apply(json.dumps)

# Save to new CSV without index
new_df.to_csv('data-preprocessing/bitagent.data/samples/bfcl_modified.csv', index=False)