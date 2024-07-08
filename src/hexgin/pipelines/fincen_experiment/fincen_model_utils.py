"""Model utilities - different functions and objects useful in preparation and training of the models."""

import torch.nn as nn
import torch_geometric as tg
import hexgin.pipelines.fincen_experiment.fincen_models as fincen_models
import hexgin.lib.models.common as cmn
import hexgin.lib.models.experimental.hexgin_model as hg
import torch_geometric.nn as tgnn

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import hexgin.consts as cc


@dataclass
class EmbeddingParams:
    """Parameters for embedding entities, filings and countries."""

    ent_embed_dim: int
    filing_embed_dim: int
    country_emebd_dim: int


@dataclass
class HexGINConvParams:
    """Params specific for the HexGIN convolution operation"""

    activation: str
    relations: Dict[str, List[Tuple[int, int]]]
    aggregation: str
    aggregation_params: Dict[str, Any]
    align_nets: Dict[str, int]
    batch_norm: bool


@dataclass
class HexGINNetParams:
    """Params for constructing the HexGIN model."""

    embedding_params: Dict[str, Any]
    conv_params: List[Dict[str, Any]]
    linkpred_dims: List[int]
    linkpred_act_f: str

    @property
    def embed_params(self) -> EmbeddingParams:
        """The dictionary of embedding params into
        EmbeddingParams object.

        Returns
        -------
        EmbeddingParams
            Embedding params object
        """
        return EmbeddingParams(**self.embedding_params)

    @property
    def conv_specs(self) -> List[HexGINConvParams]:
        """Turns a dictionary of convolution parameters into a list of
        HexGINConvParams objects.

        Returns
        -------
        List[HexGINConvParams]
            List of HexGINConvParams objects.
        """
        return [HexGINConvParams(**d) for d in self.conv_params]


@dataclass
class SAGEParams:
    """Parameters for the GraphSAGE model."""

    sage_dims: List[int]
    hidden_act_f: str
    out_act_f: str
    linpred_act_f: str
    embedding_params: EmbeddingParams

    @property
    def embed_params(self) -> EmbeddingParams:
        return EmbeddingParams(**self.embedding_params)


STR2ACTF = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "leakyReLu": nn.LeakyReLU(0.005),
}


def build_hexin_conv_layer(hexgin_conv_params: HexGINConvParams) -> hg.HexGINLayer:
    """Constructs a HexGIN convolution layer from parameters.

    Parameters
    ----------
    hexgin_conv_params : HexGINConvParams
        HexGIN convolution parameters.

    Returns
    -------
    hg.HexGINLayer
        HexGIN convolution layer.
    """
    relation_nets = {}
    align_nets = {}
    for k, vals in hexgin_conv_params.relations.items():
        s, r, o = k.split("__")
        layers = []
        for dim_in, dim_out in vals:
            if hexgin_conv_params.batch_norm:
                layers.append(nn.BatchNorm1d(dim_in))
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(STR2ACTF[hexgin_conv_params.activation])
        relation_nets[(s, r, o)] = nn.Sequential(*layers)
    for k, v in hexgin_conv_params.align_nets.items():
        align_nets[k] = nn.Sequential(nn.Linear(v, v), nn.ReLU())

    aggr = hexgin_conv_params.aggregation
    if aggr == "multi":
        aggrs = []
        for aggr in hexgin_conv_params.aggregation_params["aggrs"]:
            if aggr == "powermean":
                aggrs.append(tgnn.PowerMeanAggregation(learn=True))
            elif aggr == "softmax":
                aggrs.append(tgnn.SoftmaxAggregation(learn=True))
            else:
                aggrs.append(aggr)
            aggr_params = dict(hexgin_conv_params.aggregation_params)
            aggr_params["aggrs"] = aggrs
        aggr = tgnn.MultiAggregation(**aggr_params)
    return hg.HexGINLayer(relation_nets, align_nets, aggr=aggr)


def build_preproc_layer(
    dim_ents: int, dim_countries: int, dim_filings: int, embed_params: EmbeddingParams
) -> cmn.PreprocModel:
    """
    Builds embedding layers for the FinCen dataset.

    Parameters
    ----------
    dim_ents: int
        Dimensionality of entities - no. of unique values.
    dim_countries: int
        Dimensionality of countries - no. of unique values.
    dim_filings: int
        Dimensionality of filings - no. of unique values.
    embed_params: EmbeddingParams
        Parameters for embedding layers.

    Returns
    -------
    PrepocModel
        Prepocessing model for x_dict.
    """
    input_sizes = {
        cc.NODE_ENTITY: dim_ents + 1,
        cc.NODE_FILING: dim_filings + 1,
        cc.COUNTRY: dim_countries + 1,
    }
    embed_dims = {
        cc.NODE_ENTITY: embed_params.ent_embed_dim,
        cc.NODE_FILING: embed_params.filing_embed_dim,
        cc.COUNTRY: embed_params.country_emebd_dim,
    }
    return fincen_models.FinCENPreprocModel(input_sizes, embed_dims)


def build_linkpred_layer(dims: List[int], activation_f: str = "relu") -> nn.Module:
    """
    Builds link prediction layer - a final layer in most GNN models.

    Parameters
    ----------
    dims: List[int]
        Diemnsionalities of subsequent layers.

    activation_f: str
        Activation function to use.

    Returns
    -------
    nn.Module
        Link prediction layer.
    """
    activation_f = STR2ACTF[activation_f]
    layers = []
    for dim_in, dim_out in zip(dims[:-1], dims[1:]):
        # layers.append(nn.BatchNorm1d(dim_in)) #TODO: consider removing it later
        layers.append(nn.Linear(dim_in, dim_out))
        if dim_out != 1:  # TODO: make it nicer
            layers.append(activation_f)
    return nn.Sequential(*layers)


def build_mlp(
    dim_ents: int, dim_countries: int, dim_filings: int, mlp_params: Dict[str, Any]
) -> fincen_models.MlpModel:
    """
    Constructs MLP model from parameters and provided dimensions.

    Parameters
    ----------
    dim_ents: int
        Dimensionality of entities - no. of unique values.
    dim_countries: int
        Dimensionality of countries - no. of unique values.
    dim_filings: int
        Dimensionality of filings - no. of unique values.
    mlp_params: Dict[str, Any]
        Parameters for MLP model.

    Returns
    -------
    fincen_models.MlpModel
        MLP model.
    """
    preproc_layer = build_preproc_layer(
        dim_ents,
        dim_countries,
        dim_filings,
        EmbeddingParams(**mlp_params["embedding_params"]),
    )
    return fincen_models.MlpModel(preproc_layer, mlp_params["hdims"])


def build_graph_sage_model(
    graph: tg.data.HeteroData,
    dim_ents: int,
    dim_countries: int,
    dim_filings: int,
    sage_params: Dict[str, Any],
) -> cmn.HeteroGNNLinkPredModel:
    """
    Constructs GraphSAGE model from parameters and provided dimensions.

    Parameters
    ----------
    graph: tg.data.HeteroData
        Heterogeneous graph data.
    dim_ents: int
        Dimensionality of entities - no. of unique values.
    dim_countries: int
        Dimensionality of countries - no. of unique values.
    dim_filings: int
        Dimensionality of filings - no. of unique values.
    gdata: tg.data.HeteroData
        Heterogeneous graph data.
    sage_params: Dict[str, Any]
        Parameters for GraphSAGE model.

    Returns
    -------
    cmn.HeteroGNNLinkPredModel
        GraphSAGE model.
    """
    sage_params = SAGEParams(**sage_params)
    preproc_model = build_preproc_layer(
        dim_ents, dim_countries, dim_filings, sage_params.embed_params
    )
    linkpred_layer = build_linkpred_layer(
        [sage_params.sage_dims[-1], 1], sage_params.linpred_act_f
    )
    sage_conv = fincen_models.SAGE(
        hidden_dim=sage_params.sage_dims[0],
        out_dim=sage_params.sage_dims[1],
        h_act=STR2ACTF[sage_params.hidden_act_f],
        out_act=STR2ACTF[sage_params.out_act_f],
    )
    sage_conv_hetero = tg.nn.to_hetero(sage_conv, graph.metadata())
    return cmn.HeteroGNNLinkPredModel(
        cc.TARGET_RELATION, preproc_model, sage_conv_hetero, linkpred_layer
    )


def build_hexgin_net(
    dim_ents: int, dim_countries: int, dim_filings: int, hexgin_params: Dict[str, Any]
) -> cmn.HeteroGNNLinkPredModel:
    """Constructs a full HexGIN networks from the parameters.

    Parameters
    ----------
    dim_ents : int
        Number of unique entities.
    dim_countries : int
        Number of unique countries.
    dim_filings : int
        Number of unique filings.
    hexgin_params : Dict[str, Any]
        HexGIN parameters.

    Returns
    -------
    cmn.HeteroGNNLinkPredModel
        Link prediction model with HexGIN architecture.
    """
    hexgin_params = HexGINNetParams(**hexgin_params)
    preproc_layer = build_preproc_layer(
        dim_ents, dim_countries, dim_filings, hexgin_params.embed_params
    )
    conv_layers = []
    for conv_params in hexgin_params.conv_specs:
        conv_layers.append(build_hexin_conv_layer(conv_params))
    hexgin_model = hg.HexGINModel(conv_layers)
    linkpred_subnet = build_linkpred_layer(
        hexgin_params.linkpred_dims, hexgin_params.linkpred_act_f
    )
    hexgin_linkpred = cmn.HeteroGNNLinkPredModel(
        cc.TARGET_RELATION, preproc_layer, hexgin_model, linkpred_subnet
    )
    return hexgin_linkpred
