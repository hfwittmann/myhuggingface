"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from .pipelines import get_data
from .pipelines import tokenize_data


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    gd = get_data.create_pipeline()
    td = tokenize_data.create_pipeline()

    return {"gd": gd, "td": td, "__default__": pipeline([gd + td])}
