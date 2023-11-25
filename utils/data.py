from datasets import DatasetDict, Dataset
from typing import List, Dict
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle


def load_data_to_dataset(data_folder: str) -> DatasetDict:
    dataset_dict = DatasetDict()
    splits = ['train', 'dev', 'test']
    base_path = 'biased.word.'
    for split in splits:
        file_path = os.path.join(data_folder, base_path + split)
        data = pd.read_csv(file_path, sep='\t', header=None, names=["id", "src_tok", "tgt_tok", "src_raw", "tgt_raw", "src_POS_tags", "tgt_parse_tags"])
        dataset = Dataset.from_pandas(data)
        dataset_dict[split] = dataset.map(lambda example: {'source': example['src_raw'], 'target': example['tgt_raw']})
    return dataset_dict


def remove_duplicates(dataset: Dataset) -> Dataset:
    df = dataset.to_pandas()
    df = df.drop_duplicates(subset=['source', 'target'])
    return Dataset.from_pandas(df)


def remove_outliers(dataset: Dataset) -> Dataset:
    df = dataset.to_pandas()

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

    return Dataset.from_pandas(df)


def convert_to_classification_dataset(dataset: Dataset) -> Dataset:
    df = dataset.to_pandas()

    # Create a new DataFrame with 'text' and 'label' columns
    df_new = pd.DataFrame({
        'text': pd.concat([df['source'], df['target']]),
        'label': [1] * len(df) + [0] * len(df)
    })

    # Reorder records so biased/unbiased pairs alternate in sequence
    dfs = np.split(df_new, indices_or_sections=2, axis=0)
    dfs = [df.reset_index(drop=True) for df in dfs]
    df_new = pd.concat(dfs).sort_index(kind="merge").reset_index(drop=True)

    return Dataset.from_pandas(df_new)


def preprocess_data(file_path: str) -> DatasetDict:
    dataset_dict = load_data_to_dataset(file_path)
    for split in dataset_dict:
        dataset_dict[split] = remove_duplicates(dataset_dict[split])
        dataset_dict[split] = remove_outliers(dataset_dict[split])
        dataset_dict[split] = convert_to_classification_dataset(dataset_dict[split])
    return DatasetDict(dataset_dict)