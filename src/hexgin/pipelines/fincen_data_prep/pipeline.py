"""
Pipeline responsible for preparation of the data for the hexgin model.
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import find_common_entities, prepare_encoders_and_scaler, prepare_matrices


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=find_common_entities,
                inputs=["entities", "originators_beneficiaries", "concerned"],
                outputs="entities_subset",
                name="find_common_entities",
            ),
            node(
                func=prepare_encoders_and_scaler,
                inputs=["entities_subset", "filings"],
                outputs=[
                    "entity_encoder",
                    "filing_encoder",
                    "country_encoder",
                    "amount_scaler",
                ],
                name="prepare_encoders",
            ),
            node(
                func=prepare_matrices,
                inputs=[
                    # Data inputs
                    "entities_subset",
                    "concerned",
                    "filings",
                    "originators_beneficiaries",
                    # Encoders inputs
                    "entity_encoder",
                    "filing_encoder",
                    "country_encoder",
                ],
                outputs=[
                    "filings_matrix",
                    "originator2filing_matrix",
                    "filing2beneficiary_matrix",
                    "filing2concerned_matrix",
                    "entities_ids_matrix",
                ],
            ),
        ],
    )
