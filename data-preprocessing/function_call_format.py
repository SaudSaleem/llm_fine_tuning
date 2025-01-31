import json
import csv
import os
import ast
import re

def process_output_json(output_str):
    try:
        # Attempt 1: Use regex to extract function name and parameters directly
        func_name_match = re.search(r'"name"\s*:\s*"([^"]+)"', output_str)
        if not func_name_match:
            return None
        func_name = func_name_match.group(1)

        params = []
        # Check for arguments
        args_match = re.search(r'"arguments"\s*:\s*{([^}]+)}', output_str)
        if args_match:
            args_content = args_match.group(1)
            params = re.findall(r'"([^"]+)"\s*:', args_content)
        else:
            # Check for parameters with properties
            params_match = re.search(r'"parameters"\s*:\s*{\s*"properties"\s*:\s*{([^}]+)}', output_str)
            if params_match:
                props_content = params_match.group(1)
                params = re.findall(r'"([^"]+)"\s*:', props_content)

        return f"{func_name}({', '.join(params)})"

    except Exception as e:
        print(f"Error: {e}\nRaw Output: {output_str}")
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