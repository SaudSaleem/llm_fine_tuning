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

# Define file pairs to process
data_folder = "data-preprocessing/bitagent.data/samples"
file_pairs = [
    ("bitagent_processed.csv", "bitagent_function.csv"),
    ("glaive_processed.csv", "glaive_function.csv"),
    ("bfcl_processed.csv", "bfcl_function.csv")
]

for input_base, output_base in file_pairs:
    # Read input CSV
    input_path = os.path.join(data_folder, input_base)
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"\nProcessing {input_base}")
    print(f"Total rows found: {len(rows)}")

    # Process rows
    processed_data = []
    for row in rows:
        output_text = row['output'].strip()
        new_output = process_output_json(output_text)
        if new_output:
            processed_data.append({
                "input": row['input'],
                "output": new_output
            })

    # Save transformed data
    output_path = os.path.join(data_folder, output_base)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['input', 'output'])
        writer.writeheader()
        writer.writerows(processed_data)

    print(f"\nProcessing summary for {input_base}:")
    print(f"Valid conversations processed: {len(processed_data)}")
    print(f"Output saved to: {output_path}")
    print(f"Success rate: {len(processed_data)/len(rows):.1%}")