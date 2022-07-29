"""
This is a boilerplate pipeline 'get_data'
generated using Kedro 0.18.2
"""
from datasets import Dataset
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
import pandas as pd


def get_data_from_web():

    dataset = load_dataset("conll2003")

    d_with_split_information = dict()

    for key in dataset:
        blub = pd.DataFrame(dataset[key])
        blub["split"] = key
        d_with_split_information[key] = Dataset.from_pandas(blub)

    dataset_out = DatasetDict(d_with_split_information)

    return dataset_out  # , dataset, dataset
