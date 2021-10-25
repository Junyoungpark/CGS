from time import perf_counter

import dgl
import hydra
import torch
import torch.optim as th_op
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from CGS.experiments.porenet.generate_graph import generate_graphs_seq
from CGS.gnn.CGS.get_model import get_model
from CGS.utils.test_utils import print_perf


@hydra.main(config_path="./CGS/configs/porenet", config_name='cgs')
def main(config=None):
    device = config.train.device

    # initialize model, optimizer, loss function
    model = get_model(num_heads=config.model.num_heads,
                      gamma=config.model.gamma,
                      num_hidden_gn=config.model.num_hidden_gn,
                      nf_dim=config.model.nf_dim,
                      ef_dim=config.model.ef_dim,
                      sol_dim=config.model.sol_dim,
                      n_hidden_dim=config.model.n_hidden_dim,
                      e_hidden_dim=config.model.e_hidden_dim,
                      node_aggregator=config.model.node_aggregator,
                      non_linear=config.model.non_linear,
                      mlp_num_neurons=config.model.mlp_num_neurons,
                      reg_num_neurons=config.model.reg_num_neurons,
                      activation=config.model.activation).to(device)

    opt = getattr(th_op, config.opt.name)(model.parameters(), lr=config.opt.lr)
    scheduler = CosineAnnealingWarmRestarts(opt, T_0=32)
    loss_fn = torch.nn.MSELoss()

    # 50 ~ 200 nodes (pores)
    ns_range = [13, 87] if config.train.tessellation == 'Delaunay' else [10, 40]

    for i in range(config.train.n_updates):
        if i % config.train.generate_g_every == 0:
            train_g = generate_graphs_seq(n_graphs=config.train.bs,
                                          nS_bd=ns_range,
                                          tessellation=config.train.tessellation)
            train_g = dgl.batch(train_g).to(device)

        start = perf_counter()
        train_nf, train_ef = train_g.ndata['feat'], train_g.edata['feat']
        train_y = train_g.ndata['target']
        train_pred = model(train_g, train_nf, train_ef)

        loss = loss_fn(train_pred, train_y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        fit_time = perf_counter() - start

        # logging
        log_dict = {'iter': i,
                    'loss': loss.item(),
                    'fit_time': fit_time,
                    'forward_itr': model.fp_layer.frd_itr,
                    'lr': opt.param_groups[0]['lr']}
        print_perf(log_dict)


if __name__ == '__main__':
    main()
