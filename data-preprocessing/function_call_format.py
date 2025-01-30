import json
import csv
import os

def process_output_json(output_str):
    try:
        # Fix escaped quotes and parse JSON
        json_str = output_str.replace('""', '"').replace("'", '"')  # Handle single quotes in input
        data = json.loads(json_str)

        # Handle lists (e.g., third example)
        if isinstance(data, list):
            for item in data:
                # Skip entries with "role": "assistant"
                if "role" in item:
                    continue
                # Prioritize entries with "name" and parameters/arguments
                if "name" in item and ("parameters" in item or "arguments" in item):
                    data = item
                    break
            else:
                return None  # No valid entries found

        # Extract function name
        func_name = data["name"]

        # Extract parameters from either "parameters.properties" or "arguments"
        if "arguments" in data:
            params = list(data["arguments"].keys())
        elif "parameters" in data and "properties" in data["parameters"]:
            params = list(data["parameters"]["properties"].keys())
        else:
            params = []

        return f"{func_name}({', '.join(params)})"

    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error processing JSON: {e}")
        return None

# Read input CSV
# Read input CSV
data_folder = "bitagent.data/samples"
input_file = "bfcl_processed.csv"
with open(os.path.join(data_folder, input_file), 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Process rows
processed_data = []
for row in rows:
    input_text = row['input']
    output_text = row['output'].strip()
    
    new_output = process_output_json(output_text)
    if new_output:
        processed_data.append({
            "input": input_text,
            "output": new_output
        })

# Save transformed data
with open('bfcl_function.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['input', 'output'])
    writer.writeheader()
    writer.writerows(processed_data)
print('saved file', processed_data)