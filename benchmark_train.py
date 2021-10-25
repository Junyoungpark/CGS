from time import perf_counter

import hydra
import torch
import torch.optim as th_op
from dgl.data import GINDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from CGS.gnn.CGS.get_model import get_graphtask_model
from CGS.utils.data import GINDataLoader
from CGS.utils.test_utils import print_perf, set_seed


def eval_model(config, model, dataloader, criterion):
    model.eval()

    total = 0
    total_loss = 0
    total_correct = 0

    for data in dataloader:
        graphs, labels = data
        graphs = graphs.to(config.train.device)
        labels = labels.to(config.train.device)
        nf = graphs.ndata['attr'].float()
        ef = torch.zeros(graphs.num_edges(), 1).float().to(config.train.device)

        total += len(labels)
        with torch.no_grad():
            outputs = model(graphs, nf, ef)
        _, predicted = torch.max(outputs.data, 1)

        total_correct += (predicted == labels.data).sum().item()
        loss = criterion(outputs, labels)
        total_loss += loss.item() * len(labels)

    loss, acc = 1.0 * total_loss / total, 1.0 * total_correct / total

    model.train()

    return loss, acc


@hydra.main(config_path="./CGS/configs/benchmark", config_name='cgs')
def main(config=None):
    use_cuda = True if 'cuda' in config.train.device else False
    set_seed(config.train.seed, use_cuda)

    device = config.train.device

    # load data
    dataset = GINDataset(name=config.exp.dataset,
                         self_loop=config.train.self_loop,
                         degree_as_nlabel=config.train.degree_as_nlabel)

    train_loader, val_loader = GINDataLoader(dataset,
                                             batch_size=config.train.bs,
                                             device=torch.device(config.train.device),
                                             seed=config.train.seed,
                                             shuffle=True,
                                             split_name='fold10',
                                             fold_idx=config.exp.fold_idx).train_valid_loader()

    config.model.nf_dim = dataset.dim_nfeats  # assigning input node dimension
    config.model.sol_dim = dataset.gclasses  # assigning solution dimension

    model = get_graphtask_model(num_heads=config.model.num_heads,
                                gamma=config.model.gamma,
                                num_hidden_gn=config.model.num_hidden_gn,
                                nf_dim=config.model.nf_dim,
                                ef_dim=config.model.ef_dim,
                                sol_dim=config.model.sol_dim,
                                n_hidden_dim=config.model.n_hidden_dim,
                                e_hidden_dim=config.model.e_hidden_dim,
                                non_linear=config.model.non_linear,
                                node_readout=config.model.node_readout,
                                node_aggregator=config.model.node_aggregator,
                                mlp_num_neurons=config.model.mlp_num_neurons,
                                reg_dp=config.model.reg_dp,
                                reg_num_neurons=config.model.reg_num_neurons,
                                activation=config.model.activation).to(device)

    opt = getattr(th_op, config.opt.name)(model.parameters(), lr=config.opt.lr)
    scheduler = CosineAnnealingWarmRestarts(opt, T_0=32)
    loss_fn = torch.nn.CrossEntropyLoss()

    max_train_acc, max_val_acc = 0.0, 0.0
    iters = len(train_loader)
    for ep in range(config.train.epochs):
        for i, (train_g, train_label) in enumerate(train_loader):
            train_g = train_g.to(device)
            train_label = train_label.to(device)

            start = perf_counter()
            train_nf = train_g.ndata['attr'].float()
            train_ef = torch.zeros(train_g.num_edges(), 1).float().to(device)
            train_pred = model(train_g, train_nf, train_ef)

            loss = loss_fn(train_pred, train_label)
            opt.zero_grad()
            loss.backward()

            opt.step()
            scheduler.step(ep + i / iters)

            fit_time = perf_counter() - start

            # logging
            log_dict = {'iter': i,
                        'train_loss': loss,
                        'fit_time': fit_time,
                        'forward_itr': model.fp_layer.frd_itr,
                        'lr': opt.param_groups[0]['lr']}

            # evaluate model
            if i % config.train.eval_every == 0:
                with torch.no_grad():
                    train_loss, train_acc = eval_model(config, model, train_loader, loss_fn)
                    val_loss, val_acc = eval_model(config, model, val_loader, loss_fn)

                # report the validation score per gradient steps
                # Seems like the most standard evaluation scheme.
                # GIN paper/implementation, GraphNorm implementation, LP-GNN paper
                log_dict['train_loss'] = train_loss
                log_dict['train_acc'] = train_acc
                log_dict['val_loss'] = val_loss
                log_dict['val_acc'] = val_acc
                log_dict['epoch'] = ep

                # report the max. validation score over the training steps
                # IGNN evaluation scheme
                # Line 153 of https://github.com/SwiftieH/IGNN/blob/main/graphclassification/train_IGNN.py
                max_train_acc = train_acc if train_acc >= max_train_acc else max_train_acc
                max_val_acc = val_acc if val_acc >= max_val_acc else max_val_acc
                log_dict['max_train_acc'] = max_train_acc
                log_dict['max_val_acc'] = max_val_acc
                print_perf(log_dict)


if __name__ == '__main__':
    main()
