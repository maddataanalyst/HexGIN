"""Common model parts - shared across different architectures. Example: preproc module, that
performs embedding of ids (entities, filings, etc.)"""

import torch as th
import torch.nn as nn
import torch_geometric.data as tgd
from typing import Dict, Tuple


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_hyperparameters(self) -> dict:
        raise NotImplementedError


class PreprocModel(nn.Module):
    """
    A generic model, that performs embedding or other preprocessing on the x_dict before passing it to the
    graph convolution layers.
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        embedding_sizes: Dict[str, int],
        *args,
        **kwargs
    ):
        super().__init__()
        self.embedding_sizes = embedding_sizes
        embedding_layers = {}
        for type, embedding_dim in embedding_sizes.items():
            input_size = input_dims[type]
            embedding_layers[type] = nn.Embedding(
                num_embeddings=input_size, embedding_dim=embedding_dim
            )
        self.embedding_layers = nn.ModuleDict(embedding_layers)

    def forward(self, x_dict: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        x_dict_embed = x_dict.copy()
        for k, x in x_dict.items():
            x_dict_embed[k] = self.embedding_layers[k](x)
        return x_dict_embed


class HeteroGNNLinkPredModel(nn.Module):
    """
    Generic Heterogeneous GNN model for the link prediction task. Composed of three parts:
    1. Preprocessing layers for each node type (if needed)
    2. Message passing layer
    3. Link prediction subnet
    """

    def __init__(
        self,
        target_relation: Tuple[str, str, str],
        preproc_model: PreprocModel,
        gnn_conv: nn.Module,
        link_pred_subnet: nn.Module,
    ):
        super().__init__()
        self.target_relation = target_relation
        self.preproc_model = preproc_model
        self.gnn_conv = gnn_conv
        self.link_pred_subnet = link_pred_subnet

    def forward(self, hetero_data: tgd.HeteroData, *args, **kwargs):
        use_labeled_edges = kwargs.get("use_labeled_edges", True)
        x_dict_embed = self.preproc_model(hetero_data.x_dict)
        h = self.gnn_conv(x_dict_embed, hetero_data.edge_index_dict)

        src_type = self.target_relation[0]
        trg_type = self.target_relation[2]

        edges_dict = (
            hetero_data.edge_label_index_dict
            if use_labeled_edges
            else hetero_data.edge_index_dict
        )

        h_src = edges_dict[self.target_relation][0]
        h_trg = edges_dict[self.target_relation][1]

        prod = h[src_type][h_src] * h[trg_type][h_trg]
        return self.link_pred_subnet(prod)
