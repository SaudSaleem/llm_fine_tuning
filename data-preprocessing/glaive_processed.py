import pandas as pd
import json
import os
import re

def extract_functions(system_text):
    """Extract all JSON function definitions from system text"""
    functions = []
    buffer = str(system_text).strip()
    
    # Remove non-JSON prefix using regex
    buffer = re.sub(r'^SYSTEM:.*?{', '{', buffer, count=1, flags=re.DOTALL)
    
    while True:
        try:
            # Find the start of a JSON object
            start = buffer.find('{')
            if start == -1:
                break
                
            # Try to parse from the first '{'
            decoder = json.JSONDecoder()
            function, idx = decoder.raw_decode(buffer[start:])
            functions.append(function)
            
            # Move buffer forward
            buffer = buffer[start + idx :].lstrip()
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            break
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            break
            
    return functions

# Define paths
data_folder = "bitagent.data/samples"
input_file = "glaive_sample.csv"
output_file = "glaive_processed.csv"

# Load dataset
df = pd.read_csv(os.path.join(data_folder, input_file))

processed_data = []

for index, row in df.iterrows():
    try:
        # Changed from 'System' to 'system' to match column name
        system_text = row['system']
        functions = extract_functions(system_text)
        
        if not functions:
            print(f"No functions found in row {index}")
            print("Buffer content:")
            print(system_text)
            continue
            
        for func in functions:
            # Create input-output pair
            processed_data.append({
                'input': func['description'],
                'output': json.dumps({
                    'name': func['name'],
                    'parameters': func.get('parameters', {})
                }, indent=2)
            })
            
    except KeyError as e:
        print(f"Column error: Make sure CSV has 'system' column. {str(e)}")
        break
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
print(f"Valid functions found: {len(processed_data)}")
print(f"Output saved to: {os.path.join(data_folder, output_file)}")