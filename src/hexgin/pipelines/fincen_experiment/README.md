# HexGIN Experiment Pipeline README

This pipeline is responsible for conducting the entire experiment, which includes cross-validation of models, and training and testing of HexGIN, SAGE, and MLP models.

## Cross Validation of Models

### Function: `cross_validate_models`

This function performs a full cross-validation of the HexGIN, SAGE, and MLP models and returns the results.

#### Inputs:

- `train_graph`: Training data - a heterogeneous graph.
- `entity_encoder`: Encoder for entities.
- `filing_encoder`: Encoder for filings.
- `country_encoder`: Encoder for countries.
- `params:model_params`: Model parameters.
- `params:crossval_params`: Cross-validation parameters.
- `params:training_params`: Training parameters.

#### Outputs:

- `crossval_metrics`: Cross-validation metrics.
- `crossval_stats`: Melted metrics.
- [`bootstrap_metrics`]: Bootstraped metrics.

## HexGIN Model Training

### Function: `train_hexgin`

This function trains the HexGIN model and tests it on the test data.

#### Inputs:

- `train_graph`: Training heterogeneous graph.
- `val_graph`: Validation heterogeneous graph.
- `test_graph`: Testing heterogeneous graph.
- `entity_encoder`: Encoder for entities.
- `filing_encoder`: Encoder for filings.
- `country_encoder`: Encoder for countries.
- `params:model_params`: Model parameters.
- `params:training_params`: Training params.

#### Outputs:

- `hexgin_test_report`: A classification report for the HexGIN model.
- `hexgin_metrics`: Test metrics for the HexGIN model.

## SAGE Model Training

### Function: `train_sage`

This function trains the SAGE model and tests it on the test data.

#### Inputs:

- `train_graph`: Training heterogeneous graph.
- `val_graph`: Validation heterogeneous graph.
- `test_graph`: Testing heterogeneous graph.
- `entity_encoder`: Encoder for entities.
- `filing_encoder`: Encoder for filings.
- `country_encoder`: Encoder for countries.
- `params:model_params`: Model parameters.
- `params:training_params`: Training params.

#### Outputs:

- `sage_test_report`: A classification report for the SAGE model.
- `sage_metrics`: Test metrics for the SAGE model.

## MLP Model Training

### Function: `train_mlp`

This function trains the MLP model and tests it on the test data.

#### Inputs:

- `train_graph`: Training heterogeneous graph.
- `val_graph`: Validation heterogeneous graph.
- `test_graph`: Testing heterogeneous graph.
- `entity_encoder`: Encoder for entities.
- `filing_encoder`: Encoder for filings.
- `country_encoder`: Encoder for countries.
- `params:model_params`: Model parameters.
- `params:training_params`: Training params.
-
#### Outputs:

- `mlp_test_report`: A classification report for the MLP model.
- `mlp_metrics`: Test metrics for the MLP model.
