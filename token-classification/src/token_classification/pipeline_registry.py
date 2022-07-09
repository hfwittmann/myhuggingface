"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from .pipelines import get_data


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    gd = get_data.create_pipeline()

    return {"gd": gd, "__default__": pipeline([gd])}
