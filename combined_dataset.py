import os
import ast
import json
import random
import logging
import pandas as pd

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# MAKE DATA IN FUNCTION CALL FORMAT
try:
    processing_scripts = [
        'data-preprocessing/bitagent_processed.py',
        'data-preprocessing/bfcl_processed.py',
        'data-preprocessing/glaive_processed.py'
    ]
    
    for script in processing_scripts:
        logger.info(f"Executing processing script: {script}")
        with open(script) as file:
            exec(file.read())
    
    logger.info("All data processing steps completed successfully.")
except Exception as e:
    print(f"Error during preprocessing: {e}")
    exit()  # Stop execution if preprocessing fails



def merge_csv_files(file_paths, output_path):
    try:
        # Load and merge CSV files
        df_list = [pd.read_csv(file) for file in file_paths]
        merged_df = pd.concat(df_list, ignore_index=True)

        # Remove rows where either 'conversations' or 'tools' is empty
        merged_df = merged_df.dropna(subset=['conversation', 'tools'])
        merged_df = merged_df[
            (merged_df['conversation'].astype(str).str.strip() != '') &
            (merged_df['conversation'].astype(str) != '[]') &
            (merged_df['tools'].astype(str).str.strip() != '') &
            (merged_df['tools'].astype(str) != '[]')
        ]
        # Save the merged file
        merged_df.to_csv(output_path, index=False)

        # Return the number of rows in the merged file
        return merged_df.shape[0]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

# File paths
file_paths = [
    "data-preprocessing/bitagent.data/samples/bitagent_modified.csv",
    "data-preprocessing/bitagent.data/samples/bfcl_modified.csv",
    "data-preprocessing/bitagent.data/samples/glaive_modified.csv"
]

# Output file path
merged_path = "bitAgent.csv"

# Merge files and get the number of rows
num_rows = merge_csv_files(file_paths, merged_path)

if num_rows is not None:
    print(f"Merged file saved at {merged_path} with {num_rows} rows.")


# ADD EXTRAB TOOLS IN TOOLS COLUMNS
# Read source CSV and extract tools
modified_path = "data-preprocessing/bitagent.data/samples/bitagent_modified.csv"
modified_df = pd.read_csv(modified_path)

# Collect all tools from all rows
all_tools = []
for tools_str in modified_df['tools']:
    # print('tools_str', tools_str)
    tools = json.loads(tools_str)
    all_tools.extend(tools)

# merged_path = "data-preprocessing/bitagent.data/samples/merged.csv"
merged_df = pd.read_csv(merged_path)

errored_rows = 0
# Update tools column with JSON-safe formatting
def update_tools(tools_str):
  try:
      global errored_rows
      selected_tool = random.choice(all_tools)
      tools_list = json.loads(tools_str)
      # Determine the number of tools to add (2-6)
      num_tools_to_add = random.randint(2, 6)
      for _ in range(num_tools_to_add):
        selected_tool = random.choice(all_tools)
        # Randomly choose to append or prepend each tool
        if random.choice([True, False]):
            tools_list.append(selected_tool)
        else:
            tools_list.insert(0, selected_tool)
      return json.dumps(tools_list)
  except Exception as e:
        errored_rows += 1
        print(f"Unexpected error: {e}", tools_str)
        return tools_str

merged_df['tools'] = merged_df['tools'].apply(update_tools)

# Save modified data back to merged.csv
merged_df.to_csv(merged_path, index=False)

print(f"Successfully added tool name to merged.csv and these are errored rows", errored_rows)
