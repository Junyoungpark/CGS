import torch.nn as nn

from CGS.nn.MPNNLayer import MPNNLayer


class AttnMPNN(nn.Module):

    def __init__(self,
                 node_in_dim: int,
                 edge_in_dim: int,
                 node_hidden_dim: int = 64,
                 edge_hidden_dim: int = 64,
                 node_out_dim: int = 64,
                 edge_out_dim: int = 64,
                 num_hidden_gn: int = 0,
                 node_aggregator: str = 'mean',
                 mlp_params: dict = {},
                 preserve_cardinality: bool = False,
                 spectral_norm: bool = False):
        super(AttnMPNN, self).__init__()

        self.node_in_dim = node_in_dim
        self.node_hidden_dim = node_hidden_dim
        self.node_out_dim = node_out_dim

        self.edge_in_dim = edge_in_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.edge_out_dim = edge_out_dim
        self.preserve_cardinality = preserve_cardinality
        self.spectral_norm = spectral_norm

        node_in_dims = [node_in_dim] + num_hidden_gn * [node_hidden_dim]
        node_out_dims = num_hidden_gn * [node_hidden_dim] + [node_out_dim]

        edge_in_dims = [edge_in_dim] + num_hidden_gn * [edge_hidden_dim]
        edge_out_dims = num_hidden_gn * [edge_hidden_dim] + [edge_out_dim]

        self.layers = nn.ModuleList()
        for ni, no, ei, eo in zip(node_in_dims, node_out_dims,
                                  edge_in_dims, edge_out_dims):
            gn = MPNNLayer(node_indim=ni,
                           node_outdim=no,
                           edge_indim=ei,
                           edge_outdim=eo,
                           node_aggregator=node_aggregator,
                           **mlp_params)
            self.layers.append(gn)

    def forward(self, g, nf, ef):
        for gn in self.layers:
            nf, ef = gn(g, nf, ef)
        return nf, ef

    def __repr__(self):
        msg = '\n'
        msg += "Attention MPNN \n"
        msg += "Num GN layers : {} \n".format(len(self.layers))
        msg += "In dims : Node {} | Edge {} \n".format(self.node_in_dim, self.edge_in_dim)
        msg += "Hidden dims : Node {} | Edge {} \n".format(self.node_hidden_dim, self.edge_hidden_dim)
        msg += "Out dims : Node {} | Edge {} \n".format(self.node_out_dim, self.edge_out_dim)
        return msg
