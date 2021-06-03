import torch.nn as nn
import torch.nn.functional as F

from CGS.gnn.IGNN.IGNNLayer import ImplicitGraph
from CGS.gnn.IGNN.utils import get_spectral_rad
from CGS.nn.MLP import MLP
from CGS.nn.MPNN import AttnMPNN


class IGNN(nn.Module):

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 lifted_dim: int,  # bias function input dim
                 hidden_dim: int,  # hidden state dim (state of the fp equation)
                 output_dim: int,
                 activation: str,
                 num_hidden_gn: int,
                 mlp_num_neurons: list = [128],
                 reg_num_neurons: list = [64, 32]):
        super(IGNN, self).__init__()

        self.encoder = AttnMPNN(node_in_dim=node_dim,
                                edge_in_dim=edge_dim,
                                node_hidden_dim=64,
                                edge_hidden_dim=64,
                                node_out_dim=lifted_dim,
                                edge_out_dim=1,  # will be ignored
                                num_hidden_gn=num_hidden_gn,
                                node_aggregator='sum',
                                mlp_params={'num_neurons': mlp_num_neurons,
                                            'hidden_act': activation,
                                            'out_act': activation})

        self.ignn = ImplicitGraph(lifted_dim, hidden_dim, None, kappa=0.9)
        self.decoder = MLP(hidden_dim, output_dim,
                           hidden_act=activation,
                           num_neurons=reg_num_neurons)

    def forward(self, g, nf, ef):
        """
        1. Transform input graph with node/edge features to the bias terms of the fixed point equations
        2. Solve fixed point eq
        3. Decode the solution with MLP.
        """

        unf, _ = self.encoder(g, nf, ef)

        adj = g.adj().to(nf.device)
        adj_rho = get_spectral_rad(adj)
        z = self.ignn(None, adj, unf.T, F.relu, adj_rho, A_orig=None).T
        pred = self.decoder(z)
        return pred
