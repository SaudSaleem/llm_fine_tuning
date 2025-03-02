import requests
import json
import pandas as pd
from tqdm import tqdm
import sys
import os

# Add parent directory to path to import validate_tool_call
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reward_based_validation import validate_tool_call

# Define the URL and the headers
url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}

# Load evaluation data
eval_df = pd.read_csv('data-preprocessing/bitagent.data/samples/bitagent_shuffle_processed.csv')

# Verify required columns exist
if 'conversation' not in eval_df.columns or 'tools' not in eval_df.columns:
    raise ValueError("CSV must contain 'conversation' and 'tools' columns")

total_rows = 0
total_reward = 0
max_possible_reward = 0
valid_tool_calls = 0

for index, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
    try:
        # Parse conversation and tools
        conversation = json.loads(row['conversation'])
        tools_data = json.loads(row['tools'])
        
        # Find the tool call message in conversation
        tool_call_message = None
        for message in conversation:
            if message.get('role') == 'tool call':
                tool_call_message = message
                break
                
        if not tool_call_message:
            continue
            
        # Extract tool call JSON
        tool_call_json = json.dumps(tool_call_message)
        if not tool_call_json:
            continue
            
        total_rows += 1
        
        # Prepare messages for the model
        messages = []
        
        # Add system message with tools
        system_content = f"""You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.

You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke:
{json.dumps(tools_data, indent=2)}"""
        
        messages.append({"role": "system", "content": system_content})
        
        # Add user messages from conversation (exclude tool call and assistant messages)
        for message in conversation:
            if message.get('role') == 'user':
                messages.append({"role": "user", "content": message.get('content')})
                
        # Prepare request payload
        data = {
            "model": "saudsaleem/bitagent-autotrain",
            "messages": messages
        }
        
        # Make API call
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_data = response.json()
        
        # Get model's prediction
        model_response = response_data['choices'][0]['message']['content'].strip()
        
        # Validate the tool call
        print(f"model_response: {model_response}")
        print(f"tool_call_json: {tool_call_json}")
        validation_result = validate_tool_call(tool_call_json, model_response)
        print(f"validation_result: {validation_result}")
        # Update metrics
        total_reward += validation_result['total_reward']
        max_possible_reward += validation_result['max_reward']
        
        if validation_result['valid']:
            valid_tool_calls += 1
            
        print(f"Row {index} - Valid: {validation_result['valid']}, Reward: {validation_result['total_reward']}/{validation_result['max_reward']}")
        print(f"Feedback: {validation_result['feedback']}")
        
    except Exception as e:
        print(f"Error processing row {index}: {str(e)}")
        continue

# Calculate and print accuracy metrics
if total_rows > 0:
    reward_accuracy = (total_reward / max_possible_reward) * 100 if max_possible_reward > 0 else 0
    valid_call_accuracy = (valid_tool_calls / total_rows) * 100
    
    print(f"\nReward-based Accuracy: {reward_accuracy:.2f}%")
    print(f"Total Reward: {total_reward:.2f}/{max_possible_reward:.2f}")
    print(f"\nValid Tool Call Accuracy: {valid_call_accuracy:.2f}%")
    print(f"Valid Tool Calls: {valid_tool_calls}/{total_rows}")
else:
    print("No valid tool call rows found in the dataset")