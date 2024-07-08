"""
Pipeline that contains allsteps of the HexGIN model experiment.
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import cross_validate_models, train_hexgin, train_sage, train_mlp


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=cross_validate_models,
                inputs=[
                    "train_graph",
                    "entity_encoder",
                    "filing_encoder",
                    "country_encoder",
                    "params:model_params",
                    "params:crossval_params",
                    "params:training_params",
                ],
                outputs=["crossval_metrics", "crossval_stats", "bootstrap_metrics"],
                name="cross_validate_models",
            ),
            node(
                func=train_hexgin,
                inputs=[
                    "train_graph",
                    "val_graph",
                    "test_graph",
                    "entity_encoder",
                    "filing_encoder",
                    "country_encoder",
                    "params:model_params",
                    "params:training_params",
                ],
                outputs=["hexgin_test_report", "hexgin_metrics"],
                name="train_hexgin",
            ),
            node(
                func=train_sage,
                inputs=[
                    "train_graph",
                    "val_graph",
                    "test_graph",
                    "entity_encoder",
                    "filing_encoder",
                    "country_encoder",
                    "params:model_params",
                    "params:training_params",
                ],
                outputs=["sage_test_report", "sage_metrics"],
                name="train_sage",
            ),
            node(
                func=train_mlp,
                inputs=[
                    "train_graph",
                    "val_graph",
                    "test_graph",
                    "entity_encoder",
                    "filing_encoder",
                    "country_encoder",
                    "params:model_params",
                    "params:training_params",
                ],
                outputs=["mlp_test_report", "mlp_metrics"],
                name="train_mlp",
            ),
        ]
    )
