"""
This is a boilerplate pipeline 'experiment_prep'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import build_graph, split_graph


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_graph,
                inputs=[
                    "entities_ids_matrix",
                    "filings_matrix",
                    "originator2filing_matrix",
                    "filing2beneficiary_matrix",
                    "filing2concerned_matrix",
                ],
                outputs="hetero_graph",
                name="build_graph",
            ),
            node(
                func=split_graph,
                inputs=["hetero_graph", "params:split_params"],
                outputs=["train_graph", "val_graph", "test_graph"],
                name="split_graph",
            ),
        ]
    )
