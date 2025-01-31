import pandas as pd
import os
import logging

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
        'data-preprocessing/glaive_processed.py',
        'data-preprocessing/function_call_format.py'
    ]
    
    for script in processing_scripts:
        logger.info(f"Executing processing script: {script}")
        with open(script) as file:
            exec(file.read())
    
    logger.info("All data processing steps completed successfully.")
except Exception as e:
    print(f"Error during preprocessing: {e}")
    exit()  # Stop execution if preprocessing fails

# Define paths
data_folder = "bitagent.data/samples"
output_file = "combined_dataset.csv"
files_to_merge = [
    "glaive_function.csv",
    "bfcl_function.csv",
    "bitagent_function.csv"
]

def load_and_validate(file_path):
    """Load a CSV file with validation checks"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    
    # Check required columns
    if not {'input', 'output'}.issubset(df.columns):
        print(f"Missing columns in {file_path}")
        return None
    
    return df

# Load all datasets
datasets = []
for filename in files_to_merge:
    file_path = os.path.join(data_folder, filename)
    df = load_and_validate(file_path)
    if df is not None:
        print(f"Loaded {len(df)} rows from {filename}")
        datasets.append(df)
    else:
        print(f"Skipped {filename}")

# Merge datasets
if datasets:
    combined_df = pd.concat(datasets, ignore_index=True)
    
    # Save combined dataset
    combined_df.to_csv(
        os.path.join(data_folder, output_file),
        index=False,
        quoting=2,
        escapechar='\\',
        encoding='utf-8'
    )
    
    print("\nMerging completed successfully!")
    print(f"Total combined rows: {len(combined_df)}")
    print(f"Output saved to: {os.path.join(data_folder, output_file)}")
else:
    print("No valid datasets to merge")