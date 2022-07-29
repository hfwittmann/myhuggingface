"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from .pipelines import get_data
from .pipelines import tokenize_data
from .pipelines import model
from .pipelines import labelstudio


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    """_summary_

    Returns:
        _type_: _description_
    """  
    gd = get_data.create_pipeline()
    td = tokenize_data.create_pipeline()
    mo = model.create_pipeline()
    ls = labelstudio.create_pipeline()

    return {"gd": gd, "ls":ls, "td": td, "mo": mo, "__default__": pipeline([gd + ls+ td + mo])}
