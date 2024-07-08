# Experiment Preparation Pipeline

## Introduction
This is the second pipeline in the project, responsible for preparing the experiment for the HexGIN model. It consists of two main parts, each performing a specific function in the experiment preparation process.

# `build_graph`
This function is used to build the heterogeneous graph from the prepared matrices. It returns heterogeneus graph for the HexGIN model.

## Inputs:

1. `entities_ids_matrix`
1. `filings_matrix`
1. `originator2filing_matrix`
1. `filing2beneficiary_matrix`
1. `filing2concerned_matrix`

## Output:

`hetero_graph`

# `split_graph`
This function splits the built graph into training, validation, and testing graphs.

## Inputs:

1. `hetero_graph`
1. `params:split_params`

## Outputs:

1. `train_graph`
1. `val_graph`
1. `test_graph`