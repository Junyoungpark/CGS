import dgl
import torch.nn as nn


def get_aggregator(mode, from_field='m', to_field='agg_m'):
    AGGR_TYPES = ['sum', 'mean', 'max']
    if mode in AGGR_TYPES:
        if mode == 'sum':
            aggr = dgl.function.sum(from_field, to_field)
        if mode == 'mean':
            aggr = dgl.function.mean(from_field, to_field)
        if mode == 'max':
            aggr = dgl.function.max(from_field, to_field)
    else:
        raise RuntimeError("Given aggregation mode {} is not supported".format(mode))
    return aggr


class BasicReadout(nn.Module):
    """
    a NN module wrapper class for graph readout
    """

    def __init__(self, op):
        super(BasicReadout, self).__init__()
        self.op = op

    def forward(self, g, x):
        with g.local_scope():
            g.ndata['feat'] = x
            rd = dgl.readout_nodes(g, 'feat', op=self.op)
            return rd
