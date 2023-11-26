from datasets import DatasetDict, Dataset, load_from_disk
from typing import List, Dict
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from typing import Tuple, Dict


def display_data(dataset_dict, num_rows=3):
    """
    Display the dataset dictionary and sample rows for each split.

    Args:
        dataset_dict (dict): A dictionary containing the dataset splits.
        num_rows (int, optional): The number of sample rows to display. Defaults to 3.
    """
    for split in dataset_dict:
        print(f"\n{split} dataset:")
        print(dataset_dict[split])
        print("\nSample rows:")
        for i in range(num_rows):
            print(dataset_dict[split][i])


def load_data_to_dataset(data_folder: str) -> DatasetDict:
    """
    Loads data from the specified folder and converts it into a DatasetDict object.

    Args:
        data_folder (str): The path to the folder containing the data files.

    Returns:
        DatasetDict: A dictionary containing the loaded datasets for train, dev, and test splits.
    """
    dataset_dict = DatasetDict()
    splits = ['train', 'dev', 'test']
    base_path = 'biased.word.'
    for split in splits:
        file_path = os.path.join(data_folder, base_path + split)
        data = pd.read_csv(file_path, sep='\t', header=None, names=["id", "src_tok", "tgt_tok", "src_raw", "tgt_raw", "src_POS_tags", "tgt_parse_tags"])
        dataset = Dataset.from_pandas(data)
        dataset_dict[split] = dataset.map(lambda example: {'source': example['src_raw'], 'target': example['tgt_raw']})
    return dataset_dict


def remove_duplicates(dataset: Dataset) -> Tuple[Dataset, pd.DataFrame]:
    """
    Removes duplicate rows from the dataset based on the 'id' column.

    Args:
        dataset (Dataset): The input dataset.

    Returns:
        Tuple[Dataset, pd.DataFrame]: A tuple containing the deduplicated dataset and a DataFrame
        containing the duplicate rows that were removed.
    """
    df = dataset.to_pandas()
    df_duplicates = df[df.duplicated(subset=['id'], keep=False)]
    df = df.drop_duplicates(subset=['id'])
    return Dataset.from_pandas(df), df_duplicates


def remove_outliers(dataset: Dataset) -> Dataset:
    """
    Removes outliers from the dataset based on certain criteria.

    Args:
        dataset (Dataset): The input dataset.

    Returns:
        Tuple[Dataset, DataFrame]: A tuple containing the modified dataset and the outliers.

    """
    df = dataset.to_pandas()

    # Keep a copy of original DataFrame to find outliers later
    original_df = df.copy()

    # Remove unusually long source sentences (greater than the 99th percentile)
    upper_threshold = df['source'].str.len().quantile(0.99)
    df = df[df['source'].str.len() <= upper_threshold]

    # Remove unusually short source sentences (less than the 1st percentile)
    lower_threshold = df['source'].str.len().quantile(0.01)
    df = df[df['source'].str.len() >= lower_threshold]

    # Remove sentences with net subtraction of more than 1 word
    df = df[df['source'].str.split().str.len() - df['target'].str.split().str.len() <= 1]

    # Remove sentences with net addition of more than 4 words
    df = df[df['target'].str.split().str.len() - df['source'].str.split().str.len() <= 4]

    # Find the outliers
    outliers = original_df.loc[~original_df.index.isin(df.index)]

    # Reset the index for clean conversion to Dataset
    df = df.reset_index(drop=True)

    return Dataset.from_pandas(df), outliers


def convert_to_classification_dataset(path: str) -> DatasetDict:
    """
    Converts a saved dataset into a classification dataset.

    Args:
        path (str): The path to the saved dataset.

    Returns:
        DatasetDict: The converted classification dataset.
    """
    # Load the saved dataset
    dataset_dict = load_from_disk(path)

    new_dataset_dict = {}

    for split in dataset_dict.keys():
        df = dataset_dict[split].to_pandas()

        # Create a new DataFrame with 'text' and 'label' columns
        df_new = pd.DataFrame({
            'text': pd.concat([df['source'], df['target']]),
            'label': [1] * len(df) + [0] * len(df)
        })

        # Shuffle the DataFrame
        df_new = df_new.sample(frac=1, random_state=42).reset_index(drop=True)
        new_dataset_dict[split] = Dataset.from_pandas(df_new)

    return DatasetDict(new_dataset_dict)


def save_processed_data(dataset_dict: Dict[str, Dataset], target_path: str):
    """
    Save the processed data to disk.

    Args:
        dataset_dict (Dict[str, Dataset]): A dictionary containing the datasets for different splits.
        target_path (str): The path where the processed data will be saved.

    Returns:
        None
    """
    # Keep only the necessary columns
    for split in dataset_dict:
        dataset_dict[split] = dataset_dict[split].remove_columns(
            [col for col in dataset_dict[split].column_names if col not in ["id", "source", "target"]]
        )

    # Save the DatasetDict as Apache Arrow tables
    os.makedirs(target_path, exist_ok=True)
    dataset_dict.save_to_disk(target_path)
