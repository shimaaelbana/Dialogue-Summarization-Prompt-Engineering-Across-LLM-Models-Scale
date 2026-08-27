"""Data loading utilities for the DialogSum dataset."""
from datasets import load_dataset

DATASET_NAME = "knkarthick/dialogsum"


def load_dialogsum():
    """Load the DialogSum dataset (train/validation/test splits) from Hugging Face."""
    return load_dataset(DATASET_NAME)


def get_example(dataset, split: str, index: int):
    """Return the (dialogue, summary) pair at `index` in the given split."""
    row = dataset[split][index]
    return row["dialogue"], row["summary"]
