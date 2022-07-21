"""
This is a boilerplate pipeline 'tokenize_data'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import tokenize


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                tokenize,
                inputs="dataset_json",
                # inputs=["dataset", "dataset_csv", "dataset_json"],
                outputs="dataset_tokenized_json",
                name="tokenize",
                tags=["token"],
            )
        ]
    )
