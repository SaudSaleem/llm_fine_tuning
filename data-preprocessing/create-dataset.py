import os
import logging
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download
import pandas as pd
from itertools import islice

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ShuffledJSONDatasetIterator:
    def __init__(self):
        self.dataframes = []
        self.all_data = None
        self.shuffled_data = None
        self.index = 0

        for filename in ["java", "javascript", "simple", "multiple", "sql", "live_simple", "live_multiple"]:
            bfcl_path = f"bitagent.data/bfcl/BFCL_v3_{filename}.json"
            bfcl_answer_path = f"bitagent.data/bfcl/possible_answer/BFCL_v3_{filename}.json"
            if os.path.exists(bfcl_path) and os.path.exists(bfcl_answer_path):
                df_data = pd.read_json(bfcl_path, lines=True)
                df_answer = pd.read_json(bfcl_answer_path, lines=True)
                df_data['ground_truth'] = df_answer['ground_truth']
                self.dataframes.append(df_data[['id', 'question', 'function', 'ground_truth']])
        self.all_data = pd.concat(self.dataframes)
        self._shuffle_data()

    def _shuffle_data(self):
        self.shuffled_data = self.all_data.sample(frac=1).reset_index(drop=True)
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.shuffled_data):
            row = self.shuffled_data.iloc[self.index]
            self.index += 1
            return row
        else:
            self._shuffle_data()
            return self.__next__()


def huggingface_loader(dataset_name, root_data_dir="bitagent.data", split="train", name=None):
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


def load_bfcl_dataset(dataset_name, root_data_dir="bitagent.data", split="train", name=None):
    snapshot_download(
        repo_id=dataset_name,
        allow_patterns="*.json",
        repo_type="dataset",
        local_dir="bitagent.data/bfcl/"
    )
    return ShuffledJSONDatasetIterator()


def sample_and_save_datasets(output_dir="bitagent.data/samples", sample_size=1000):
    os.makedirs(output_dir, exist_ok=True)

    try:
        glaive_ds = huggingface_loader("glaiveai/glaive-function-calling-v2")
        glaive_df = pd.DataFrame(glaive_ds)
        glaive_sample = glaive_df.sample(n=min(sample_size, len(glaive_df)))
        glaive_sample.to_csv(f"{output_dir}/glaive_sample.csv", index=False)
        logger.info(f"Saved Glaive sample to {output_dir}/glaive_sample.csv")
    except Exception as e:
        logger.error(f"Error processing Glaive dataset: {str(e)}")

    try:
        bitagent_ds = huggingface_loader("BitAgent/tool_calling")
        bitagent_df = pd.DataFrame(bitagent_ds)
        bitagent_sample = bitagent_df.sample(n=min(sample_size, len(bitagent_df)))
        bitagent_sample.to_csv(f"{output_dir}/bitagent_sample.csv", index=False)
        logger.info(f"Saved BitAgent sample to {output_dir}/bitagent_sample.csv")
    except Exception as e:
        logger.error(f"Error processing BitAgent dataset: {str(e)}")

    try:
        bfcl_ds = load_bfcl_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard")
        bfcl_sample = pd.DataFrame(list(islice(bfcl_ds, sample_size)))
        bfcl_sample.to_csv(f"{output_dir}/bfcl_sample.csv", index=False)
        logger.info(f"Saved BFCL sample to {output_dir}/bfcl_sample.csv")
    except Exception as e:
        logger.error(f"Error processing BFCL dataset: {str(e)}")


def merge_datasets(data_folder="bitagent.data/samples"):
    bfcl = pd.read_csv(os.path.join(data_folder, "bfcl_sample.csv"))
    bitagent = pd.read_csv(os.path.join(data_folder, "bitagent_sample.csv"))
    glaive = pd.read_csv(os.path.join(data_folder, "glaive_sample.csv"))

    bfcl['input'] = bfcl['question']
    bfcl['output'] = bfcl['ground_truth']
    bfcl = bfcl[['input', 'output']]

    bitagent['input'] = bitagent['conversation']
    bitagent['output'] = bitagent['tools']
    bitagent = bitagent[['input', 'output']]

    glaive['input'] = glaive['system'] + "\n" + glaive['chat']
    glaive['output'] = ""
    glaive = glaive[['input', 'output']]

    combined_dataset = pd.concat([bfcl, bitagent, glaive], ignore_index=True)
    merged_file_path = os.path.join(data_folder, "merged_dataset.csv")
    combined_dataset.to_csv(merged_file_path, index=False)
    logger.info(f"Merged dataset saved as: {merged_file_path}")
    return merged_file_path


if __name__ == "__main__":
    sample_and_save_datasets()
    merged_path = merge_datasets()
    print(f"Merged dataset saved at: {merged_path}")
