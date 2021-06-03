from functools import partial

import dgl
import torch
import numpy as np


def compute_q_dgl(edges, pressure_field, q_field):
    pi = edges.src[pressure_field]
    pj = edges.dst[pressure_field]
    hydraulic_conductance = edges.data['k']
    q = hydraulic_conductance * (pi - pj)
    return {q_field: q}


def compute_delta_p_dgl(edges, pressure_field, delta_p_field):
    pi = edges.src[pressure_field]
    pj = edges.dst[pressure_field]
    return {delta_p_field: pi - pj}


def compute_q(g, pressure, pressure_field='_p', q_field='_q'):
    g.ndata[pressure_field] = pressure
    g.apply_edges(func=partial(compute_q_dgl,
                               pressure_field=pressure_field,
                               q_field=q_field))
    _ = g.ndata.pop(pressure_field)
    return g.edata.pop(q_field)


def compute_delta_p(g, pressure, pressure_field='_p', delta_p_field='_delta_p'):
    g.ndata[pressure_field] = pressure
    g.apply_edges(func=partial(compute_delta_p_dgl,
                               pressure_field=pressure_field,
                               delta_p_field=delta_p_field))
    _ = g.ndata.pop(pressure_field)
    return g.edata.pop(delta_p_field)


def compute_edge_sum(g, ef, ef_field):
    g.edata[ef_field] = ef
    message_func = dgl.function.copy_edge(ef_field, 'm')
    reduce_func = dgl.function.sum('m', 'sum_m')
    g.pull(g.nodes(), message_func, reduce_func)
    _ = g.edata.pop(ef_field)
    return g.ndata.pop('sum_m')


def compute_q_ij_plus_q_ji(g, ef, ef_field):
    g.edata[ef_field] = ef
    src, dst = g.edges()[0], g.edges()[1]
    q_ij_plus_q_ji = g.edges[src, dst].data[ef_field] + g.edges[dst, src].data[ef_field]
    _ = g.edata.pop(ef_field)
    return q_ij_plus_q_ji


def compute_edge_feature(edges):
    src_diameter = edges.src['diameter']
    dst_diameter = edges.dst['diameter']
    pore_diameters = torch.cat([src_diameter, dst_diameter], dim=-1)
    diameter, _ = torch.min(pore_diameters, dim=-1)
    diameter = diameter.view(-1, 1).float()

    src_radii = 0.5 * src_diameter
    dst_radii = 0.5 * dst_diameter

    src_coord = edges.src['coords']
    dst_coord = edges.dst['coords']
    dist_bet_centers = torch.norm(src_coord - dst_coord, dim=1, keepdim=True)
    length = dist_bet_centers - (src_radii + dst_radii)
    length = length.float()

    volume = np.pi * (0.5 * diameter) ** 2 * length
    volume = volume.float()

    mu = edges.src['mu']
    hydraulic_conductance_numer = torch.tensor(np.pi) * (diameter * 0.5) ** 4
    hydraulic_conductance_denom = 8.0 * mu * length
    hydraulic_conductance = hydraulic_conductance_numer / hydraulic_conductance_denom

    return {'diameter': diameter,
            'length': length,
            'volume': volume,
            'k': hydraulic_conductance}


def _get_overlapped_bool(pore_diameter, pore_ij, core_distance, eps=0.0):
    pore_radii = pore_diameter * 0.5
    pore_radii_ij = pore_radii[pore_ij]
    sum_radii = np.sum(pore_radii_ij, axis=1)
    is_overlapped = core_distance - sum_radii < eps
    return is_overlapped


def get_valid_diameters(pn, pore_diameter, min_throat_length=0.0, shrinking_factor=0.1):
    pore_ij = pn['throat.conns']
    src_coord = pn['pore.coords'][pore_ij][:, 0, :]
    dst_coord = pn['pore.coords'][pore_ij][:, 1, :]
    core_distance = np.linalg.norm(src_coord - dst_coord, axis=-1)

    while True:
        is_overlapped = _get_overlapped_bool(pore_diameter,
                                             pore_ij,
                                             core_distance,
                                             eps=min_throat_length)
        if is_overlapped.sum() >= 1.0:
            overlay_pore_idx = np.unique(pn['throat.conns'][is_overlapped].flatten())
            pore_diameter[overlay_pore_idx] *= shrinking_factor
        else:
            break

    return pore_diameter
