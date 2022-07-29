"""
This is a boilerplate pipeline 'labelstudio'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import convert_annotations_to_labelstudio


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                convert_annotations_to_labelstudio,
                inputs="dataset_json",
                outputs="dataset_labelstudio",
                name="convert_annotations_to_labelstudio",
                tags=["mlops"],
            )
        ]
    )
