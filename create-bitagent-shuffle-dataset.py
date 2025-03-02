import os
import logging
import json
import random
import pandas as pd
from datasets import load_dataset, load_from_disk

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def huggingface_loader(dataset_name, root_data_dir="data-preprocessing/bitagent.data", split="train", name=None):
    logger.debug(f"Loading {dataset_name}")
    dataset_dir = f"{root_data_dir}/{dataset_name.replace('/', '_')}"
    if os.path.exists(f"{dataset_dir}/state.json"):
        logger.debug(f"Loading from disk ({dataset_dir}) ...")
        ds = load_from_disk(dataset_dir)
    else:
        logger.debug("Loading from web ...")
        ds = load_dataset(dataset_name, split=split, name=name, token=os.getenv("HF_TOKEN", None))
        ds.save_to_disk(dataset_dir)
    logger.debug("Loaded.")
    return ds

def add_extra_tool_calls():
   # ADD EXTRAB TOOLS IN TOOLS COLUMNS
    csv_path = "data-preprocessing/bitagent.data/samples/bitagent_shuffle_sample.csv"
    modified_csv_path = "data-preprocessing/bitagent.data/samples/bitagent_shuffle_processed.csv"
    modified_df = pd.read_csv(csv_path)

    # Collect all tools from all rows
    all_tools = []
    for tools_str in modified_df['tools']:
        # print('tools_str', tools_str)
        tools = json.loads(tools_str)
        all_tools.extend(tools)

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
  
    modified_df['tools'] = modified_df['tools'].apply(update_tools)
    modified_df.to_csv(modified_csv_path, index=False)
    print(f"Successfully added tool name to bitagent_shuffle_processed.csv and these are errored rows", errored_rows)



def create_bitagent_shuffle_dataset():
    try:
        bitagent_ds = huggingface_loader("BitAgent/tool_shuffle_small_test")
        bitagent_df = pd.DataFrame(bitagent_ds)
        bitagent_sample = bitagent_df.sample(frac=1)
        bitagent_sample.to_csv(f"data-preprocessing/bitagent.data/samples/bitagent_shuffle_sample.csv", index=False)
        logger.info(f"Saved BitAgent sample to data-preprocessing/bitagent.data/samples/bitagent_shuffle_sample.csv")
        add_extra_tool_calls()
    except Exception as e:
        logger.error(f"Error processing BitAgent dataset: {str(e)}")

if __name__ == "__main__":
    create_bitagent_shuffle_dataset()