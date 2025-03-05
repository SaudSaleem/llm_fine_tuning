import pandas as pd
from datasets import load_dataset, DatasetDict, load_from_disk

DATA_PATH = "data/subset_dataset"
CSV_DATA_PATH = "data/csv"
dataset = load_dataset("saudsaleem/mistral-7b-instruct-templated-dataset")

random_indices = dataset["train"].shuffle(seed=42).select(range(3000))
# Split into train (80%) and test (20%) sets
train_test_split = random_indices.train_test_split(test_size=0.2, seed=42)
subset_dataset = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})
subset_dataset.save_to_disk(DATA_PATH)

# Load the dataset from disk
dataset = load_from_disk(DATA_PATH)

# Convert train and test splits to CSV with only "text" column
train_df = pd.DataFrame(dataset["train"]["text"], columns=["text"])
test_df = pd.DataFrame(dataset["test"]["text"], columns=["text"])

# Save as CSV files in the specified path
train_df.to_csv(f"{CSV_DATA_PATH}/train.csv", index=False)
test_df.to_csv(f"{CSV_DATA_PATH}/test.csv", index=False)

print("CSV files saved successfully!")