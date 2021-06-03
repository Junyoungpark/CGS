from functools import partial

import dgl
import numpy as np
import torch

from CGS.experiments.porenet.graph_gen_utils.feature_computing_utils import (compute_delta_p,
                                                                             compute_edge_feature,
                                                                             compute_q_dgl)
from CGS.experiments.porenet.graph_gen_utils.openopm_utils import pn_to_networkx


def pn_to_dgl(pn,
              pressure: torch.tensor,
              mu: float,
              nf_keys: list,
              ef_keys: list,
              verbose=False):
    nxg = pn_to_networkx(pn)  # return the networkx graph with only edge connectivity.
    g = dgl.from_networkx(nxg)

    # setup node features
    g.ndata['coords'] = torch.tensor(pn['pore.coords'])
    g.ndata['volume'] = torch.tensor(pn['pore.volume']).view(-1, 1)
    g.ndata['diameter'] = torch.tensor(pn['pore.diameter']).view(-1, 1)

    # normalizing 'pressures' to be in the range of [-1.0, 1.0]
    # Based on the maximum principle, the maximum pressure will be given on the boundary
    # Therefore we can always compute inverse transform of this normalization.

    p_original = torch.tensor(pressure).view(-1, 1).float()
    p_max = p_original.max().float()
    g.ndata['pressure_original'] = p_original
    g.ndata['pressure_scaler'] = torch.ones_like(p_original) * p_max
    g.ndata['pressure'] = g.ndata['pressure_original'].clone() / g.ndata['pressure_scaler'].clone()
    g.ndata['mu'] = torch.ones_like(p_original) * mu

    is_bd = torch.zeros(g.number_of_nodes()).bool()
    for surface in ['front', 'back']:
        is_bd[pn.pores(surface)] = True

    g.ndata['boundary'] = is_bd.view(-1, 1).float()
    g.ndata['pressure_masked'] = g.ndata['pressure'] * g.ndata['boundary']

    # compute edge features on fly
    g.apply_edges(compute_edge_feature)
    g.apply_edges(func=partial(compute_q_dgl,
                               pressure_field='pressure_original',
                               q_field='q_original'))

    l = g.edata['length']
    r = 0.5 * g.edata['diameter']
    l_min = l.min()
    r_max = r.max()

    q_scaler_const = torch.tensor(np.pi) / (8.0 * mu)
    q_scaler = q_scaler_const * (r_max ** 4) / l_min * p_max

    q_original = g.edata['q_original']
    g.edata['q_scaler'] = torch.ones_like(q_original) * q_scaler
    g.edata['pressure_scaler'] = torch.ones_like(q_original) * p_max
    g.edata['q'] = g.edata['q_original'].clone() / g.edata['q_scaler'].clone()
    g.edata['delta_p'] = compute_delta_p(g, g.ndata['pressure'])
    g.edata['delta_p_original'] = compute_delta_p(g, g.ndata['pressure_original'])

    # Assigning global features
    g.edata['r_max'] = torch.ones_like(q_original) * r_max
    g.edata['l_min'] = torch.ones_like(q_original) * l_min
    g.ndata['r_max'] = torch.ones_like(p_original) * r_max
    g.ndata['l_min'] = torch.ones_like(p_original) * l_min
    g.ndata['q_scaler'] = torch.ones_like(p_original) * q_scaler

    if verbose:
        bd_ratio = is_bd.sum().float() / is_bd.shape[0]
        print("bd ratio : {}".format(bd_ratio))

    # set node, edge feature
    g.ndata['feat'] = torch.cat([g.ndata[k] for k in nf_keys], dim=-1).float()
    g.edata['feat'] = torch.cat([g.edata[k] for k in ef_keys], dim=-1).float()
    g.ndata['target'] = g.ndata['pressure'].float()
    return g
