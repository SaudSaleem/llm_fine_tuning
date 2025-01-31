# import pandas as pd
# import os
# from ast import literal_eval

# # Define paths
# data_folder = "bitagent.data/samples"
# input_file = "bfcl_sample.csv"
# output_file = "bfcl_processed.csv"

# # Load the dataset
# bfcl = pd.read_csv(os.path.join(data_folder, input_file))

# # Process each row to extract input and output
# processed_data = []
# for index, row in bfcl.iterrows():
#     try:
#         # Parse question and function columns
#         question_messages = literal_eval(row['question'])
#         function_data = literal_eval(row['ground_truth'])
        
#         # Extract user messages
#         user_content = []
#         for message_group in question_messages:
#             for msg in message_group:
#                 if msg['role'] == 'user':
#                     user_content.append(msg['content'])
        
#         # Create input-output pairs
#         processed_data.append({
#             'input': '\n'.join(user_content),
#             'output': str(function_data)  # Keep as string representation
#         })
        
#     except Exception as e:
#         print(f"Error processing row {index}: {str(e)}")
#         continue

# # Convert to DataFrame and save as CSV
# processed_df = pd.DataFrame(processed_data)
# processed_df.to_csv(
#     os.path.join(data_folder, output_file),
#     index=False,
#     quoting=2  # Quote all non-numeric values
# )

# print(f"Processed {len(processed_df)}/{len(bfcl)} rows successfully")
# print(f"CSV output saved to: {os.path.join(data_folder, output_file)}")



import pandas as pd
import os
from ast import literal_eval

# Define paths
data_folder = "data-preprocessing/bitagent.data/samples"
input_file = "bfcl_sample.csv"
output_file = "bfcl_processed.csv"

# Load the dataset
bfcl = pd.read_csv(os.path.join(data_folder, input_file))

def transform_function_data(function_data, index):
    """
    Convert function data to structured format with 'name' and 'parameters'.
    """
    try:
        parsed_data = literal_eval(function_data)  # Convert string to Python object
        if not isinstance(parsed_data, list) or len(parsed_data) == 0:
            print(f"⚠️ Skipping row {index}: function_data is not a list → {function_data}")
            return None
        
        func_dict = parsed_data[0]  # Extract first function dictionary
        if not isinstance(func_dict, dict):
            print(f"⚠️ Skipping row {index}: function_data[0] is not a dictionary → {func_dict}")
            return None
        
        func_name = list(func_dict.keys())[0]  # Extract function name
        params = func_dict[func_name]  # Extract parameters
        
        if not isinstance(params, dict):
            print(f"⚠️ Skipping row {index}: Parameters are not a dictionary → {params}")
            return None

        # Convert parameters into structured format
        structured_params = {
            "type": "object",
            "properties": {},
            # "required": list(params.keys())  # Set required parameters
        }

        for key, value in params.items():
            if not isinstance(value, list) or len(value) == 0:
                print(f"⚠️ Skipping row {index}: Invalid parameter format → {key}: {value}")
                return None
            
            structured_params["properties"][key] = {
                "type": "number" if isinstance(value[0], (int, float)) else "string",
                "description": f"Parameter for {key}"
            }
        
        return {
            "name": func_name,
            "parameters": structured_params
        }
    
    except Exception as e:
        print(f"⚠️ Error processing function_data at index {index}: {function_data} → {str(e)}")
        return None

# Process each row to extract input and output
processed_data = []
for index, row in bfcl.iterrows():
    try:
        # Parse question and function columns
        question_messages = literal_eval(row['question'])
        function_data = row['ground_truth']

        # Extract user messages
        user_content = []
        for message_group in question_messages:
            for msg in message_group:
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    user_content.append(msg.get('content', ''))

        # Transform function data
        structured_output = transform_function_data(function_data, index)

        # Create input-output pairs
        if structured_output:
            processed_data.append({
                'input': '\n'.join(user_content),
                'output': str(structured_output)  # Convert dict to string for CSV
            })

    except Exception as e:
        print(f"⚠️ Error processing row {index}: {str(e)}")
        continue

# Convert to DataFrame and save as CSV
processed_df = pd.DataFrame(processed_data)
processed_df.to_csv(
    os.path.join(data_folder, output_file),
    index=False,
    quoting=2  # Quote all non-numeric values
)

print(f"Processed {len(processed_df)}/{len(bfcl)} rows successfully")
print(f"CSV output saved to: {os.path.join(data_folder, output_file)}")
