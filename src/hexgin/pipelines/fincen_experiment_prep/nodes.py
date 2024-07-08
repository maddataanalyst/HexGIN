"""The experiment preparation code for the HexGIN model."""

from typing import Tuple

import torch as th
import torch_geometric as tg
import torch_geometric.data as tgdata
import numpy as np

import hexgin.consts as cc


def build_graph(
    entity_ids_matrix: np.ndarray,
    filings_matrix: np.ndarray,
    originator2filing_matrix: np.ndarray,
    filing2beneficiary_matrix: np.ndarray,
    filing2concerned_matrix: np.ndarray,
) -> tgdata.HeteroData:
    """
    Build a heterogeneous graph from the matrices.

    Parameters
    ----------
    entity_ids_matrix: np.ndarray
        The entities ids matrix
    filings_matrix: np.ndarray
        The filings matrix
    originator2filing_matrix: np.ndarray
        The originator2filing matrix: origintor entity id to filing id
    filing2beneficiary_matrix: np.ndarray
        The filing2beneficiary matrix: filing id to beneficiary entity id
    filing2concerned_matrix: np.ndarray
        The filing2concerned matrix: filing id to concerned entity id

    Returns
    -------
    tgdata.HeteroData
        The heterogeneous graph
    """
    gdata = tgdata.HeteroData()
    gdata[cc.NODE_ENTITY].x = th.tensor(entity_ids_matrix, dtype=th.long).squeeze()
    gdata[cc.NODE_FILING].x = th.tensor(filings_matrix, dtype=th.long).squeeze()

    gdata["entity", "sends", "filing"].edge_index = th.tensor(
        originator2filing_matrix, dtype=th.long
    )
    gdata["filing", "benefits", "entity"].edge_index = th.tensor(
        filing2beneficiary_matrix, dtype=th.long
    )
    gdata["filing", "concerns", "entity"].edge_index = th.tensor(
        filing2concerned_matrix, dtype=th.long
    )

    return gdata


def split_graph(
    graph: tgdata.HeteroData, split_params: dict
) -> Tuple[tgdata.HeteroData, tgdata.HeteroData, tgdata.HeteroData]:
    """Given a heterogeneous graph, split it into train, validation, and test sets using the
    RandomLinkSplit transformer.

    Parameters
    ----------
    graph : tgdata.HeteroData
        Hetero graph to split
    split_params : dict
        Splitting params like num_val, num_test, neg_sampling_ratio, etc.

    Returns
    -------
    Tuple[tgdata.HeteroData, tgdata.HeteroData, tgdata.HeteroData]
        A tuple of:
        1. Train data
        2. Validation data
        3. Test data
    """
    train_test_val_split_seed = split_params[cc.CFG_TRAIN_TEST_SPLIT_SEED]
    tg.seed_everything(train_test_val_split_seed)

    splitter = tg.transforms.RandomLinkSplit(
        num_val=split_params[cc.CFG_NUM_VAL],
        num_test=split_params[cc.CFG_NUM_TEST],
        neg_sampling_ratio=split_params[cc.CFG_NEG_SAMPLING_RATIO],
        add_negative_train_samples=split_params[cc.CFG_ADD_NEGATIVE_TRAIN_SAMPLES],
        disjoint_train_ratio=split_params[cc.CFG_DISJOINT_TRAIN_RATIO],
        edge_types=cc.TARGET_RELATION,
    )
    train_data, val_data, test_data = splitter(graph)
    return train_data, val_data, test_data
