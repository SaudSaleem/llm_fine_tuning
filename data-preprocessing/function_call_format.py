import json
import csv
import os 
def convert_json_to_function_call(output_json_str):
    try:
        # Fix escaped quotes and parse JSON
        output_json = json.loads(output_json_str.replace('""', '"'))
        
        # Extract function name
        func_name = output_json["name"]
        
        # Extract parameters from "properties" keys (fallback if "required" is missing)
        parameters = output_json["parameters"]["properties"].keys()
        params_str = ", ".join(parameters)
        
        return f"{func_name}({params_str})"
    except KeyError as e:
        print(f"Key error: {e} in JSON: {output_json_str}")
        return None

# Read input CSV
data_folder = "bitagent.data/samples"
input_file = "bfcl_sample.csv"
with open(os.path.join(data_folder, input_file), 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Process rows
processed_data = []
for row in rows:
    input_text = row['input']
    output_text = row['output']
    
    new_output = convert_json_to_function_call(output_text)
    if new_output:
        processed_data.append({"input": input_text, "output": new_output})

# Save transformed data
with open('transformed_data.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['input', 'output'])
    writer.writeheader()
    writer.writerows(processed_data)