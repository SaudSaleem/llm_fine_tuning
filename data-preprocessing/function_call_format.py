import json
import csv
import os
import ast 

def process_output_json(output_str):
    try:
        # Attempt 1: Standard JSON parsing
        json_str = output_str.replace('""', '"').replace("'", '"').strip()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Attempt 2: Use ast.literal_eval as fallback
            try:
                data = ast.literal_eval(json_str)
            except (SyntaxError, ValueError) as e:
                print(f"JSON Parse Failed: {e}\nRaw String: {json_str}")
                return None

        # Handle lists (e.g., third example)
        if isinstance(data, list):
            valid_item = None
            for item in data:
                if "role" in item:
                    continue
                if "name" in item and ("parameters" in item or "arguments" in item):
                    valid_item = item
                    break
            if valid_item is None:
                return None
            data = valid_item

        # Check if data is None after list processing
        if data is None:
            return None

        # Extract function name
        func_name = data["name"]

        # Extract parameters
        if "arguments" in data:
            params = list(data["arguments"].keys())
        elif "parameters" in data and "properties" in data["parameters"]:
            params = list(data["parameters"]["properties"].keys())
        else:
            params = []

        return f"{func_name}({', '.join(params)})"

    except (KeyError, TypeError, AttributeError) as e:
        print(f"Error: {e}\nData: {data}\nRaw Output: {output_str}")
        return None
# Read input CSV
# Read input CSV
data_folder = "bitagent.data/samples"
input_file = "bitagent_processed.csv"
with open(os.path.join(data_folder, input_file), 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f"Total rows processed: {len(rows)}")
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
output_file = os.path.join(data_folder, 'bitagent_function.csv')
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['input', 'output'])
    writer.writeheader()
    writer.writerows(processed_data)

print(f"\nProcessing summary:")
print(f"Valid conversations processed: {len(processed_data)}")
print(f"Output saved to: {os.path.join(data_folder, output_file)}")