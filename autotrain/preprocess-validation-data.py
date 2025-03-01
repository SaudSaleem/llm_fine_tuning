import pandas as pd
import ast
import json

# Read the CSV file
df = pd.read_csv('evaluation_data.csv')

# Function to convert invalid JSON strings to valid JSON
def fix_json(json_str):
    try:
        # Parse the string into a Python dict (handles single quotes)
        data = ast.literal_eval(json_str)
        # Convert the dict to a valid JSON string
        return json.dumps(data)
    except Exception as e:
        print(f"Failed to parse: {e}")
        return None

# Assuming your original JSON-like data is in a column named 'invalid_json'
df['JSON'] = df['JSON'].apply(fix_json)

# Save the updated CSV
df.to_csv('evaluation_data_with_valid_json.csv', index=False)