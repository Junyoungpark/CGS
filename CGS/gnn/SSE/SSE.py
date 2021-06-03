import torch
import torch.nn as nn

from CGS.gnn.SSE.SSELayer import SSELayer
from CGS.nn.MLP import MLP


class SSE(nn.Module):

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 target_dim: int,
                 activation: str = 'ReLU',
                 alpha: float = 0.1,  # damping parameter
                 tol: float = 1e-5,
                 max_iter: int = 20):

        super(SSE, self).__init__()
        self.alpha = alpha  # damping parameter for fixed point iteration
        self.tol = tol
        self.max_iter = max_iter

        self.hidden_dim = hidden_dim

        self.sse_layer = SSELayer(node_dim=node_dim,
                                  edge_dim=edge_dim,
                                  hidden_dim=hidden_dim,
                                  activation=activation)

        self.regressor = MLP(input_dim=hidden_dim,
                             output_dim=target_dim)
        self.frd_itr = None

    def solve_fp_eq(self, g, nf, ef):
        with torch.no_grad():
            h = torch.zeros(g.number_of_nodes(),
                            self.hidden_dim,
                            device=g.device)  # initial hidden
            itr = 0

            while itr < self.max_iter:
                h_next = self.sse_layer(g, h, nf, ef)
                h_next = (1 - self.alpha) * h + self.alpha * h_next  # damping
                _g = h - h_next
                if torch.norm(_g) < self.tol:
                    break
                h = h_next
                itr += 1
            return h, itr

    def forward(self, g, nf, ef):
        h, self.frd_itr = self.solve_fp_eq(g, nf, ef)  # don't track gradient.

        # re-engage autograd
        h = self.sse_layer(g, h, nf, ef)
        out = self.regressor(h)
        return out
