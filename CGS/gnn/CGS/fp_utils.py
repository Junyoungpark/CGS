import torch
from dgl.nn.functional import edge_softmax


def prepare_attn_A_b(g, A_logit, b):
    """
    :param g: DGL.graph - possibly batched
    :param A_logit: expected size [#. total edges x #. heads]
    :param b: expected size [#. total nodes x #.heads]
    :return:
        A : [#. heads x # total nodes x # total nodes]
        b : [#. heads x # total nodes x 1]
    """
    n_nodes, nh = b.shape

    # prepare A
    A_vals = edge_softmax(g, A_logit, norm_by='src')
    i, j = g.adj()._indices()
    A = torch.zeros(nh, n_nodes, n_nodes, device=A_logit.device)
    A[:, i, j] = A_vals.transpose(1, 0)

    # prepare B
    b = b.transpose(0, 1).unsqueeze(dim=-1)

    return A, b


def prepare_degree_A_b(g, A_logit, b):
    with g.local_scope():
        g.ndata['in_degree'] = g.in_degrees()
        g.ndata['out_degree'] = g.out_degrees()

        def get_normalizer(edges):
            normalizer = edges.src['out_degree']
            return {'norm': normalizer}

        g.apply_edges(get_normalizer)

        g.edata['norm']
        n_nodes, nh = b.shape

        A_logit_clipped = A_logit.clip(0.0, 1.0)
        A_vals = A_logit_clipped / g.edata['norm'].view(-1, 1)
        i, j = g.adj()._indices()
        A = torch.zeros(nh, n_nodes, n_nodes, device=A_logit.device)
        A[:, i, j] = A_vals.transpose(1, 0)

        # prepare B
        b = b.transpose(0, 1).unsqueeze(dim=-1)

        return A, b
