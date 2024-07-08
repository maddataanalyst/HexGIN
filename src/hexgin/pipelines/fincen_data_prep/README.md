# Data Preparation Pipeline
## Introduction

This is the first pipeline in the project, responsible for preparing the data for the HexGIN model. It consists of three main parts, each performing a specific function in the data preparation process.

# `find_common_entities`
This function is used to find common entities across different data sources and merges them together.

## Inputs:

1. `entities` - all entities in the data sources.
2. `originators_beneficiaries` - originators and beneficiaries in the data sources.
3. `concerned` - concerned parties for each transaction.

## Output:

`entities_subset` - subset of common entitites.

# `prepare_encoders_and_scaler`

This function prepares the encoders and scaler needed for the data transformation process.

## Inputs:

`entities_subset`
`filings`

## Outputs:

1. `entity_encoder`
2. `filing_encoder`
3. `country_encoder`
4. `amount_scaler`

# `prepare_matrices`

This function prepares the matrices required for the HexGIN model. Outputs matrices required to build a graph later on.

## Inputs:

1. `entities_subset`
1. `concerned`
1. `filings`
1. `originators_beneficiaries`
1. `entity_encoder`
1. `filing_encoder`
1. `country_encoder`

## Outputs:

Two types of matrices are returned:
1. Matrices that describe entities and their features (nodes of the graph)
2. Matrices that describe relationships between entities (edges of the graph)

3. `filings_matrix`
4. `originator2filing_matrix`
5. `filing2beneficiary_matrix`
6. `filing2concerned_matrix`
7. `entities_ids_matrix`