"""
This is a boilerplate pipeline 'get_data'
generated using Kedro 0.18.2
"""
from datasets import load_dataset


def get_data_from_web():

    dataset = load_dataset("conll2003")
    return dataset # , dataset, dataset
