from CGS.gnn.CGS.CGS import CGS, GraphTaskCGS
from CGS.nn.MLP import MLP
from CGS.nn.MPNN import AttnMPNN
from CGS.utils.gnn_utils import BasicReadout


def get_model(num_heads: int,
              gamma: float,
              num_hidden_gn: int,
              nf_dim: int,
              ef_dim: int,
              sol_dim: int,
              n_hidden_dim: int = 64,
              e_hidden_dim: int = 64,
              non_linear: str = 'Identity',
              mlp_dp: float = 0.0,
              mlp_num_neurons: list = [64, 64],
              reg_dp: float = 0.0,
              reg_num_neurons: list = [],
              activation: str = 'ReLU',
              node_aggregator: str = 'mean'):
    mlp_num_neurons = list(mlp_num_neurons)
    reg_num_neurons = list(reg_num_neurons)

    enc = AttnMPNN(node_in_dim=nf_dim,
                   edge_in_dim=ef_dim,
                   node_hidden_dim=n_hidden_dim,
                   edge_hidden_dim=e_hidden_dim,
                   node_out_dim=num_heads,
                   edge_out_dim=num_heads,
                   num_hidden_gn=num_hidden_gn,
                   node_aggregator=node_aggregator,
                   mlp_params={'dropout_prob': mlp_dp,
                               'num_neurons': mlp_num_neurons,
                               'hidden_act': activation,
                               'out_act': activation})

    dec = MLP(input_dim=num_heads,
              num_neurons=reg_num_neurons,
              hidden_act=activation,
              output_dim=sol_dim,
              dropout_prob=reg_dp)

    cgs = CGS(encoder=enc,
              decoder=dec,
              gamma=gamma,
              non_linear=non_linear)

    return cgs


def get_graphtask_model(num_heads: int,
                        gamma: float,
                        num_hidden_gn: int,
                        nf_dim: int,
                        ef_dim: int,
                        sol_dim: int,
                        n_hidden_dim: int = 64,
                        e_hidden_dim: int = 64,
                        non_linear: str = 'Identity',
                        node_readout='sum',
                        mlp_dp: float = 0.0,
                        mlp_num_neurons: list = [64, 64],
                        reg_dp: float = 0.0,
                        reg_num_neurons: list = [],
                        activation: str = 'ReLU',
                        node_aggregator: str = 'mean'):
    mlp_num_neurons = list(mlp_num_neurons)
    reg_num_neurons = list(reg_num_neurons)

    enc = AttnMPNN(node_in_dim=nf_dim,
                   edge_in_dim=ef_dim,
                   node_hidden_dim=n_hidden_dim,
                   edge_hidden_dim=e_hidden_dim,
                   node_out_dim=num_heads,
                   edge_out_dim=num_heads,
                   num_hidden_gn=num_hidden_gn,
                   node_aggregator=node_aggregator,
                   mlp_params={'dropout_prob': mlp_dp,
                               'num_neurons': mlp_num_neurons,
                               'hidden_act': activation,
                               'out_act': activation})

    readout = BasicReadout(op=node_readout)
    dec = MLP(input_dim=num_heads,
              num_neurons=reg_num_neurons,
              hidden_act=activation,
              output_dim=sol_dim,
              dropout_prob=reg_dp)

    cgs = GraphTaskCGS(encoder=enc,
                       readout=readout,
                       decoder=dec,
                       gamma=gamma,
                       non_linear=non_linear)

    return cgs
