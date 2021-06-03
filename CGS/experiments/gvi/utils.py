import dgl
import torch


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


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
