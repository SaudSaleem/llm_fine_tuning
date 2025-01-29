import pandas as pd
import os
from ast import literal_eval

# Define paths
data_folder = "bitagent.data/samples"
input_file = "bfcl_sample.csv"
output_file = "bfcl_processed.csv"

# Load the dataset
bfcl = pd.read_csv(os.path.join(data_folder, input_file))

# Process each row to extract input and output
processed_data = []
for index, row in bfcl.iterrows():
    try:
        # Parse question and function columns
        question_messages = literal_eval(row['question'])
        function_data = literal_eval(row['ground_truth'])
        
        # Extract user messages
        user_content = []
        for message_group in question_messages:
            for msg in message_group:
                if msg['role'] == 'user':
                    user_content.append(msg['content'])
        
        # Create input-output pairs
        processed_data.append({
            'input': '\n'.join(user_content),
            'output': str(function_data)  # Keep as string representation
        })
        
    except Exception as e:
        print(f"Error processing row {index}: {str(e)}")
        continue

# Convert to DataFrame and save as CSV
processed_df = pd.DataFrame(processed_data)
processed_df.to_csv(
    os.path.join(data_folder, output_file),
    index=False,
    quoting=2  # Quote all non-numeric values
)

print(f"Processed {len(processed_df)}/{len(bfcl)} rows successfully")
print(f"CSV output saved to: {os.path.join(data_folder, output_file)}")