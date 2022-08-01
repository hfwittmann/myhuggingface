"""
This is a boilerplate pipeline 'get_data'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import get_data_from_web
from .nodes import get_texts, preprocess_and_split, split_information


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # node(
            #     get_data_from_web,
            #     inputs=None,
            #     outputs= "dataset_json", # ["dataset", "dataset_csv", "dataset_json"],
            #     name="get_data_from_web",
            #     tags=["data"],
            # ),
            node(
                get_texts,
                inputs="annotations_json",
                outputs="annotations_and_texts_json",
                name="get_texts",
                tags=["data"],
            ),
            node(
                preprocess_and_split,
                inputs="annotations_and_texts_json",
                outputs="dataset_json",
                name="preprocess_and_split",
                tags=["data"],
            ),
            node(
                split_information,
                inputs="dataset_json",
                outputs="dataset_json_plus_split_information",
                name="split_information",
                tags=["data"],
            ),
        ]
    )
