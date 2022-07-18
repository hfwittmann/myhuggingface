"""
This is a boilerplate pipeline 'get_data'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import get_data_from_web


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                get_data_from_web,
                inputs=None,
                outputs= "dataset_json", # ["dataset", "dataset_csv", "dataset_json"],
                name="get_data_from_web",
                tags=["data"],
            )
        ]
    )
