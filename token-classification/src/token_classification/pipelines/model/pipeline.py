"""
This is a boilerplate pipeline 'model'
generated using Kedro 0.18.2
"""

from black import out
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import setup, train, evaluate
from .nodes import load_language_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(load_language_model, inputs="parameters", outputs="LM", name="load_language_model"),
            node(
                setup,
                inputs=[
                    "LM",
                    "dataset_tokenized_json",
                    "params:sample_train",
                    "params:sample_validation",
                    "params:model",
                ],
                outputs=["model", "evaluation"],
                name="setup",
                tags=["model", "trainer", "evaluation"],
            ),
            # node(train, inputs="trainer", outputs="trained_trainer", name="train"),
            # node(evaluate, inputs="trained_trainer", outputs="evaluation", name="evaluate"),
            # node(savemodel, inputs="trained_trainer", outputs="model", name="savemodel"),
        ]
    )
