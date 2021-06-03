import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn

from CGS.nn.MLP import MLP
from CGS.utils.gnn_utils import get_aggregator


class MPNNLayer(nn.Module):
    def __init__(self,
                 edge_indim: int,
                 edge_outdim: int,
                 node_indim: int,
                 node_outdim: int,
                 node_aggregator: str = 'mean',
                 **mlp_params):
        super(MPNNLayer, self).__init__()
        self.edge_model = MLP(input_dim=edge_indim + 2 * node_indim,
                              output_dim=edge_outdim,
                              **mlp_params)
        self.node_model = MLP(input_dim=edge_outdim + node_indim,
                              output_dim=node_outdim,
                              **mlp_params)

        self.attn_model = MLP(input_dim=edge_indim + 2 * node_indim,
                              output_dim=1,
                              **mlp_params)
        self.node_aggr = get_aggregator(node_aggregator)

    def forward(self,
                g: dgl.DGLGraph,
                nf: torch.Tensor,
                ef: torch.Tensor):
        with g.local_scope():
            g.ndata['h'] = nf
            g.edata['h'] = ef

            # perform edge update
            g.apply_edges(func=self.edge_update)

            # compute attention score
            g.edata['attn'] = dglnn.edge_softmax(g, self.attn_model(g.edata['em_input']))

            # update nodes
            g.update_all(message_func=self.message_func,
                         reduce_func=self.node_aggr,
                         apply_node_func=self.node_update)

            updated_ef = g.edata['uh']
            updated_nf = g.ndata['uh']
            return updated_nf, updated_ef

    def edge_update(self, edges):
        sender_nf = edges.src['h']
        receiver_nf = edges.dst['h']
        ef = edges.data['h']
        em_input = torch.cat([ef, sender_nf, receiver_nf], dim=-1)
        updated_ef = self.edge_model(em_input)
        return {'uh': updated_ef, 'em_input': em_input}

    @staticmethod
    def message_func(edges):
        return {'m': edges.data['uh'] * edges.data['attn']}

    def node_update(self, nodes):
        agg_m = nodes.data['agg_m']
        nf = nodes.data['h']
        nm_input = torch.cat([agg_m, nf], dim=-1)
        updated_nf = self.node_model(nm_input)
        return {'uh': updated_nf}
