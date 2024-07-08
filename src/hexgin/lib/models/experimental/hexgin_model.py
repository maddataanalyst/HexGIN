"""Main part of the research project: a HexGIN architecture combining a GIN model with a generic heterogeneous processing."""

import torch as th
import torch.nn as nn
import torch_geometric.nn as tgnn

from typing import Dict, Tuple, List


class HexGINConv(tgnn.GINConv):
    """
    This is a proposed GIN model extension that can be used with the Heterogeneous datasets. It is based on the
    implementation of the GINConv from the PyTorch Geometric library, and replaces the addition function in the
    forward pass with concatenation of neighbors vectors with the central node vector - this is especially usefull in
    heterogeneous graphs where, each entity might have a different dimensionality of embeddings.
    Additionally - HexGIN uses multiple aggregations for better generalization.
    """

    def __init__(self, subnet, align_net, *args, **kwargs):
        super().__init__(nn=subnet, *args, **kwargs)
        self.align_net = align_net

    def forward(
        self, x: th.Tensor, edge_index: th.Tensor, size=None, *args, **kwargs
    ) -> th.Tensor:
        # Part of Equation (2): x is an incoming input tensor: eigther x in first iteration or phi(hu) in the next iterations
        if isinstance(x, th.Tensor):
            x = self.align_net(x)
            x = (x, x)
        else:
            x = (self.align_net(x[0]), x[1])

        # Part of the equation (3) from paper: mv = || AGG(m_ur)
        mv = self.propagate(edge_index, x=x, size=size)

        x_v = x[1]
        if x_v is not None:
            # Part of the equation (4) from paper: (1+eps) * hv || mv
            x_v_eps = (1 + self.eps) * x_v
            mv = th.cat([x_v_eps, mv], dim=1)

        return self.nn(mv)


def out_features_from_subnet(subnet: nn.Module):
    submodules = list(subnet.modules())
    for submod in submodules[::-1]:
        if hasattr(submod, "out_features"):
            return submod.out_features


class HexGINLayer(nn.Module):
    """
    A full layer (single step of convolution) of HexGIN - intended to process
    multiple possible node types and relationships.
    """

    def __init__(
        self,
        target_dims: Dict[Tuple[str, str, str], nn.Module],
        align_nets: Dict[str, nn.Module],
        aggr: object = "sum",
    ):
        """
        Build HexGIN layer using provided subnetworks for each node type and relationship.

        Parameters
        ----------
        target_dims: Dict[Tuple[str, str, str], nn.Module]
            A dictionary mapping (src, rel, trgt) tuples to subnetworks that will be used to process.
        aggr: object
            Aggregation method to use in the final aggregation step.
        """
        super().__init__()
        self.rel_convs = {}
        self.target_dimensions = {}

        # Construct dedicated subnetworks for each relationship: implementing equations 2-4 from the paper
        for (src, rel, trgt), subnet in target_dims.items():
            align_net = align_nets[src]
            # Build HexGIN layer for relation r: HexGIN(phi_r, AGG_r)
            rel_conv = HexGINConv(subnet, align_net, train_eps=True, aggr=aggr)
            self.rel_convs[(src, rel, trgt)] = rel_conv

            self._validate_target_dims(trgt, subnet)

        # After last layer sum specific sub-networks outputs:
        # hv = SUM({hv_r) for all r in R })
        self.rel_convs = tgnn.HeteroConv(self.rel_convs, aggr="sum")

    def _validate_target_dims(self, trgt: str, subnet: nn.Module):
        """
        Validate that the target dimensionality of the subnetwork is the same as the one of the previously seen
        subnetworks for the same target.

        Parameters
        ----------
        trgt: str
            Target node type.
        subnet: nn.Module
            Subnetwork to validate.

        Raises
        ------
        ValueError
            If the target dimensionality of the subnetwork is different than the one of the previously seen
            subnetworks for the same target.
        """
        out_features = out_features_from_subnet(subnet)
        if trgt not in self.target_dimensions:
            self.target_dimensions[trgt] = out_features
        else:
            known_target_dims = self.target_dimensions[trgt]
            if known_target_dims != out_features:
                raise ValueError(
                    f"Target dimension must be the same across subnetworks. Got {known_target_dims} and {out_features} for {trgt}."
                )

    def forward(
        self,
        x_dict: Dict[str, th.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], th.Tensor],
    ):
        return self.rel_convs(x_dict, edge_index_dict)


class HexGINModel(nn.Module):
    """
    This model can contain multi-step HexGIN convolutions.
    """

    def __init__(self, hexgin_layers: List[HexGINLayer]):
        super().__init__()
        self.hexgin_layers = nn.ModuleList(hexgin_layers)

    def forward(
        self,
        x_dict: Dict[str, th.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], th.Tensor],
    ):
        """
        Performs forward pass using HexGIN convolutions.

        Parameters
        ----------
        x_dict: Dict[str, th.Tensor]
            Dictionary mapping node types to their features.
        edge_index_dict: Dict[Tuple[str, str, str], th.Tensor]
            Dictionary mapping (src, rel, trgt) tuples to edge indices.

        Returns
        -------
        h: Dict[str, th.Tensor]
            Dictionary mapping node types to their features after the forward pass.

        """
        h = x_dict
        for layer in self.hexgin_layers:
            h = layer(h, edge_index_dict)
        return h
