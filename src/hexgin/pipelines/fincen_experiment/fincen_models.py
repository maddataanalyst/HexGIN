"""Module contains the code of all 'classic' models used in the study: excluding the HexGIN model, moved to the
experimental section."""

import torch as th
import torch_geometric.data as tgdata
import torch_geometric.nn as tgnn
import torch.nn as nn

import hexgin.consts as cc
import hexgin.lib.models.common as cmn

from typing import Tuple, Dict


class FinCENPreprocModel(cmn.PreprocModel):
    def __init__(self, input_sizes: Dict[str, int], embed_dims: Dict[str, int]):
        super().__init__(input_sizes, embed_dims)

    def forward(self, x_dict: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        """Performs embedding for FinCEN files case study.
        The data for this problem has been prepared in a slightly different way:
        node type 'entity' contains two columns in its x values: col 0 - is
        entity id, and col 1 - is country id.
        Tehrefore Embedding model for this type of data is different from the
        generic one. When it encounters the 'entity' node type, it splits the
        x values into two parts, and embeds them separately, and then concatenates
        the results.


        Parameters
        ----------
        x_dict : Dict[str, th.Tensor]
            X dict for the FinCEN dataset.

        Returns
        -------
        Dict[str, th.Tensor]
            New x_dict with embedded values.
        """
        x_dict_embed = x_dict.copy()
        for node_type, x in x_dict.items():
            if node_type == "entity":
                entity_id = x_dict_embed["entity"][:, 0]
                country_id = x_dict_embed["entity"][:, 1]

                entity_embed = self.embedding_layers["entity"](entity_id)
                country_embed = self.embedding_layers["country"](country_id)
                entity_embed = th.cat((entity_embed, country_embed), dim=1)
                x_dict_embed["entity"] = entity_embed
            elif node_type == "country":
                continue
            else:
                x_dict_embed[node_type] = self.embedding_layers[node_type](x)
        return x_dict_embed


class SAGE(nn.Module):
    """
    SAGE Convolution wrapper, that can be used with the heterogeneous graphs.
    """

    def __init__(
        self, hidden_dim: int, out_dim: int, h_act=nn.ReLU(), out_act=nn.ReLU()
    ):
        super().__init__()
        self.batch_norm1 = tgnn.BatchNorm(hidden_dim)
        self.batch_norm2 = tgnn.BatchNorm(out_dim)
        self.conv1 = tgnn.SAGEConv((-1, -1), hidden_dim, aggr="sum")
        self.conv2 = tgnn.SAGEConv((-1, -1), out_dim, aggr="sum")
        self.hidden_act = h_act
        self.out_act = out_act

    def forward(self, x, edge_index):
        out1 = self.hidden_act(self.conv1(x, edge_index))
        out1norm = self.batch_norm1(out1)
        out2 = self.hidden_act(self.conv2(out1norm, edge_index))
        out2norm = self.batch_norm2(out2)
        out = self.out_act(out2norm)
        return out


class SAGEModel(cmn.HeteroGNNLinkPredModel):
    """
    A classic GNN model, that utilizes the SAGE convolution for the link prediction task on the FinCen dataset.
    """

    def __init__(
        self,
        preproc_model: FinCENPreprocModel,
        target_rel: Tuple[str, str, str] = cc.TARGET_RELATION,
        sage_dims: Tuple[int, ...] = (64, 32),
    ):
        sage = SAGE(*sage_dims)
        link_pred_f = nn.Linear(in_features=sage_dims[-1], out_features=1)
        super().__init__(target_rel, preproc_model, sage, link_pred_f)


class MlpModel(nn.Module):
    """A classic Multi-Layer perceptron model, that operates on the flattened, tabular form of a dataset"""

    def __init__(self, embed_model: cmn.PreprocModel, hdims: Tuple[int, ...] = ()):
        super().__init__()
        self.embed_model = embed_model
        self.hdims = hdims
        last_inp_size = (
            self.embed_model.embedding_sizes[cc.NODE_ENTITY]
            + self.embed_model.embedding_sizes[cc.COUNTRY]
            + self.embed_model.embedding_sizes[cc.NODE_FILING]
        )
        h_layers = []
        for hdim in hdims:
            batch_norm = nn.BatchNorm1d(last_inp_size)
            layer = nn.Linear(in_features=last_inp_size, out_features=hdim)
            h_layers.append(batch_norm)
            h_layers.append(layer)
            h_layers.append(nn.LeakyReLU(0.05))
            last_inp_size = hdim
        h_layers.append(nn.BatchNorm1d(last_inp_size))
        h_layers.append(nn.Linear(in_features=last_inp_size, out_features=1))
        self.h_layers = nn.Sequential(*h_layers)

    def forward(self, batch: tgdata.HeteroData, *args, **kwargs):
        idx_ent = batch[cc.TARGET_RELATION].edge_label_index[1, :]
        idx_filing = batch[cc.TARGET_RELATION].edge_label_index[0, :]
        x_embed: Dict[str, th.Tensor] = self.embed_model(batch.x_dict)

        ents = th.index_select(x_embed[cc.NODE_ENTITY], 0, idx_ent)
        filings = th.index_select(x_embed[cc.NODE_FILING], 0, idx_filing)

        all_embed = th.cat(
            [
                ents,
                filings,
            ],
            dim=1,
        )
        out = self.h_layers(all_embed)
        return out
