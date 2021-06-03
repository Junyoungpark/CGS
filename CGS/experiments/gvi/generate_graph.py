from functools import partial

import dgl
import numpy as np
import ray
import torch

from CGS.utils.mp_utils import chunks


@ray.remote(num_cpus=1)
def remote_generate_graph(nas, nss):
    gs = []
    for na, ns in zip(nas, nss):
        g = generate_graph(na, ns)
        gs.append(g)
    return gs


def generate_graphs(n_graphs, nA_bd, nS_bd, n_workers):
    assert nA_bd[1] < nS_bd[0], "Max #. actions requires to be smaller than min #.stats"

    nA = np.random.randint(nA_bd[0], nA_bd[1], size=n_graphs)
    nS = np.random.randint(nS_bd[0], nS_bd[1], size=n_graphs)
    nA_chunk, nS_chunk = chunks(nA, n_workers), chunks(nS, n_workers)

    gs = [remote_generate_graph.remote(nas, nss) for nas, nss in zip(nA_chunk, nS_chunk)]
    gs = [g for sub_gs in ray.get(gs) for g in sub_gs]
    return gs


def generate_graphs_seq(n_graphs, nA_bd, nS_bd):
    nA = np.random.randint(nA_bd[0], nA_bd[1], size=n_graphs)
    nS = np.random.randint(nS_bd[0], nS_bd[1], size=n_graphs)
    gs = []
    for na, ns in zip(nA, nS):
        g = generate_graph(na, ns)
        gs.append(g)
    return gs


def generate_graph(nA: int, nS: int, reward_bnd=[-1, 1], gamma=0.9, n_iters=200, atol=1e-3):
    assert nS - 1 > nA, "(nS-1) must be greater than nA"

    # sequential attach edges
    u, v = [], []
    for i in range(nS):
        srcs = list(range(nS))
        _ = srcs.pop(i)  # ignore self edges
        src = list(np.random.choice(srcs, nA))

        dst = [i] * nA

        u.extend(src)
        v.extend(dst)

    g = dgl.graph((torch.tensor(u), torch.tensor(v)), num_nodes=nS)

    r = np.random.uniform(low=reward_bnd[0], high=reward_bnd[1], size=(g.number_of_edges(), 1))
    g.edata['r'] = torch.tensor(r).float()

    g, converge_step = value_iteration(g, gamma, n_iters, atol=atol)

    # setup features
    g.ndata['feat'] = torch.ones(g.number_of_nodes(), 1)  # dummy feature
    g.edata['feat'] = g.edata['r']
    g.ndata['target'] = g.ndata['value'].clone()

    return g


def compute_action_value(edges, gamma, value_key):
    r = edges.data['r']
    value = edges.src[value_key]
    return {'action_value': r + gamma * value}


def get_max(nodes, value_key, policy_key):
    value, index = nodes.mailbox['action_value'].max(dim=1)
    return {value_key: value, policy_key: index}


def dgl_vi_backup(g, gamma, value_key='value', policy_key='policy'):
    g.pull(g.nodes(),
           message_func=partial(compute_action_value, gamma=gamma, value_key=value_key),
           reduce_func=partial(get_max, value_key=value_key, policy_key=policy_key))


def value_iteration(g, gamma=0.9, n_iters=200, value_init=None, atol=1e-3):
    if value_init is None:
        g.ndata['value'] = torch.zeros(g.number_of_nodes(), 1)
    else:
        g.ndata['value'] = value_init

    converge_step = -1
    for i in range(n_iters):
        val_prev = g.ndata['value']
        dgl_vi_backup(g, gamma=gamma)
        val_next = g.ndata['value']

        if torch.allclose(val_next, val_prev, atol=atol):
            converge_step = i
            break

    assert converge_step >= 0, "VI doesn't converge."
    return g, converge_step


def get_policy(g, value, gamma=0.9):
    with g.local_scope():
        g.ndata['_value'] = value
        dgl_vi_backup(g, gamma, value_key='_value', policy_key='_policy')
        return g.ndata.pop('_policy')
