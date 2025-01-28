import pandas as pd
import json
import os
from ast import literal_eval

# Define paths
data_folder = "bitagent.data/samples"
input_file = "bitagent_sample.csv"
output_file = "bitagent_processed.csv"

# Load dataset
df = pd.read_csv(os.path.join(data_folder, input_file))

processed_data = []

for index, row in df.iterrows():
    try:
        # Parse conversation from string to list of dicts
        conversation = literal_eval(row['conversation'])
        
        # Find user message and subsequent entries
        user_content = None
        output_entries = []
        
        for i, entry in enumerate(conversation):
            if entry['role'] == 'user':
                user_content = entry['content']
                # Process subsequent entries after user message
                for subsequent in conversation[i+1:]:
                    if subsequent['role'] == 'tool call':
                        output_entries.append(subsequent['content'])
                    else:
                        output_entries.append(subsequent)
                break
                
        if not user_content:
            print(f"No user message found in row {index}")
            continue
            
        processed_data.append({
            'input': user_content,
            'output': json.dumps(output_entries, indent=2)
        })
        
    except Exception as e:
        print(f"Error processing row {index}: {str(e)}")
        continue

# Save to CSV
processed_df = pd.DataFrame(processed_data)
processed_df.to_csv(
    os.path.join(data_folder, output_file),
    index=False,
    quoting=2,  # Quote all fields
    escapechar='\\',
    encoding='utf-8'
)

print(f"\nProcessing summary:")
print(f"Total rows processed: {len(df)}")
print(f"Valid conversations processed: {len(processed_data)}")
print(f"Output saved to: {os.path.join(data_folder, output_file)}")