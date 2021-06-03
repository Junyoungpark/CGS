import random

import dgl
import numpy as np
import torch

from CGS.experiments.gvi.generate_graph import get_policy
from CGS.utils.mp_utils import chunks


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def set_seed(seed: int,
             use_cuda: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate_model(g_list, model, device, bs=64, **kwargs):
    preds = []
    with torch.no_grad():
        for sub_gs in chunks(g_list, bs):
            sub_gs = dgl.batch(sub_gs).to(device)
            sub_pred = model(sub_gs, sub_gs.ndata['feat'], sub_gs.edata['feat'], **kwargs)
            if len(sub_pred) == 2:
                sub_pred = sub_pred[0]
            preds.append(sub_pred)
        return torch.cat(preds, dim=0).to('cpu')


def get_policy_acc(graph, values):
    """ compute the accuracy of policy predictions (per graph averaged manner)
    :param graph: dgl.graph; possibly batched
    :param values: (predicted) state values
    :return:
    """
    policy = get_policy(graph, values)
    with graph.local_scope():
        graph.ndata['correct'] = (graph.ndata['policy'] == policy).float()
        accs = dgl.readout_nodes(graph, 'correct', op='mean')  # [batch size x 1]
    mean = accs.mean().item()
    std = accs.std().item()
    return mean, std, accs


def get_pred_mape(graph, prediction, err_tol=1e-6):
    with graph.local_scope():
        y_true = graph.ndata['target']
        y_pred = prediction
        graph.ndata['ae'] = ((y_true - y_pred) / y_pred + err_tol).abs()  # absolute error
        mae = dgl.readout_nodes(graph, 'ae', op='mean')  # [batch size x 1]
        mape = mae * 100.0
    mean = mape.mean().item()
    std = mape.std().item()
    se = std / graph.batch_size  # standard error
    return mean, std, se, mape


def get_pred_mse(graph, prediction):
    with graph.local_scope():
        y_true = graph.ndata['target']
        y_pred = prediction
        graph.ndata['se'] = (y_true - y_pred) ** 2  # square error
        mse = dgl.readout_nodes(graph, 'se', op='mean')  # [batch size x 1]
    mean = mse.mean().item()
    std = mse.std().item()
    se = std / graph.batch_size  # standard error

    return_dict = {
        'mean': mean,
        'std': std,
        'se': se,
        'mse': mse
    }
    return return_dict


def print_perf(log_dict: dict):
    msg = ''
    for k, v in log_dict.items():
        msg += '{} : '.format(k)
        if isinstance(v, int):
            msg += '{} | '.format(v)
        else:
            msg += '{:.5f} | '.format(v)
    print(msg)
