import dgl
import torch
import torch.nn as nn


class SSELayer(nn.Module):
    """
    DGL implementation for Stochastic Steady state Embedding layer.
    Adopted from https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/8_sse_mx.html
    """

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 activation: str = 'ReLU'):
        super(SSELayer, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        self.W1 = nn.Linear(in_features=hidden_dim + 2 * node_dim + edge_dim,
                            out_features=hidden_dim,
                            bias=False)

        self.W2 = nn.Linear(in_features=hidden_dim,
                            out_features=hidden_dim,
                            bias=False)

        self.act = getattr(nn, activation)()

    def forward(self,
                g: dgl.DGLGraph,
                h: torch.tensor,
                nf: torch.tensor,
                ef: torch.tensor):
        def msg_func(edges):
            x = edges.dst['x']
            e = edges.data['x']
            h = edges.src['h']
            return {'m': torch.cat([x, e, h], dim=-1)}

        def reduce_func(nodes):
            m = nodes.mailbox['m'].sum(dim=1)
            z = torch.cat([nodes.data['x'], m], dim=-1)
            return {'h': self.W2(self.act(self.W1(z)))}

        with g.local_scope():
            g.ndata['x'] = nf
            g.ndata['h'] = h
            g.edata['x'] = ef
            g.update_all(msg_func, reduce_func)
            return g.ndata['h']
