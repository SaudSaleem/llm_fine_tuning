import requests
import json
import pandas as pd
from tqdm import tqdm

# Define the URL and the headers
url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}


# Load evaluation data
eval_df = pd.read_csv('evaluation_data_with_valid_json.csv')

# Verify required columns exist
if 'JSON' not in eval_df.columns:
    raise ValueError("CSV must contain 'JSON' column")

correct = 0
valid_rows = 0  # Tracks rows with valid messages

for index, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
    try:
        # Parse messages from JSON column
        messages_data = json.loads(row['JSON'])
        print('messages_data', messages_data)
        # Skip rows without messages key
        if 'messages' not in messages_data:
            continue
            
        valid_rows += 1  # Only count rows with messages

        # Process tools if present
        if 'tools' in messages_data:
            # Find or create system message
            system_msg = next((m for m in messages_data['messages'] if m['role'] == 'system'), None)
            if not system_msg:
                system_msg = {"role": "system", "content": ""}
                messages_data['messages'].insert(0, system_msg)
            
            # Append tools to system message
            tools_list = messages_data['tools']
            system_msg['content'] = f"""You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.

You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke:
{json.dumps(tools_list, indent=2)}"""
            
            # Remove tools key
            del messages_data['tools']
        print('messages_data', messages_data)
        # Prepare request payload
        data = {
            "model": "saudsaleem/bitagent-autotrain",
            "messages": messages_data['messages']
        }
        
        # Make API call
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_data = response.json()
        
        # Get model's prediction
        prediction = response_data['choices'][0]['message']['content'].strip()
        
        # Get ground truth - look in messages for assistant response
        ground_truth = messages_data.get('response', '').strip()
        
        # Compare prediction with ground truth
        if prediction.lower() == ground_truth.lower():
            correct += 1
            
    except Exception as e:
        print(f"Error processing row {index}: {str(e)}")
        continue

# Calculate and print accuracy using only valid rows
if valid_rows > 0:
    accuracy = (correct / valid_rows) * 100
    print(f"\nModel Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{valid_rows}")
else:
    print("No valid rows with 'messages' key found in the dataset")
