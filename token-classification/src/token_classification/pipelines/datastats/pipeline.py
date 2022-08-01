"""
This is a boilerplate pipeline 'datastats'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import calc_stats


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                calc_stats,
                inputs=["dataset_json", "params:datasetname"],
                outputs=None,
                name="calc_stats",
                tags=["data", "stats"],
            )
        ]
    )
