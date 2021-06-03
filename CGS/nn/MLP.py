from typing import List

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_neurons: List[int] = [64, 32],
                 hidden_act: str = 'ReLU',
                 out_act: str = 'Identity',
                 dropout_prob: float = 0.0,
                 input_norm: bool = False):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_act = getattr(nn, hidden_act)()
        self.out_act = getattr(nn, out_act)()

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        self.lins = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            self.lins.append(nn.Linear(in_dim, out_dim))

        self.dropout_p = dropout_prob

        if input_norm:
            self.input_norm = nn.BatchNorm1d(input_dim)
            self._in = True
        else:
            self._in = False

    def forward(self, xs):
        if self._in:
            xs = self.input_norm(xs)

        for i, lin in enumerate(self.lins[:-1]):
            xs = lin(xs)
            xs = self.hidden_act(xs)
            xs = F.dropout(xs, p=self.dropout_p, training=self.training)

        xs = self.lins[-1](xs)
        xs = self.out_act(xs)
        xs = F.dropout(xs, p=self.dropout_p, training=self.training)
        return xs
