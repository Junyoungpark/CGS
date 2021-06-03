import dgl
import torch
import torch.nn as nn

from CGS.gnn.CGS.fixedpoint import FixedPointLayer
from CGS.gnn.CGS.fp_utils import prepare_degree_A_b


class CGS(nn.Module):

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 gamma: float,
                 non_linear: str = None,
                 tol: float = 1e-6,
                 max_iter: int = 50):
        super(CGS, self).__init__()

        self.enc = encoder
        self.fp_layer = FixedPointLayer(gamma=gamma,
                                        activation=non_linear,
                                        tol=tol,
                                        max_iter=max_iter)

        self.dec = decoder

        self.gamma = gamma

    def forward(self, g: dgl.graph, nf: torch.tensor, ef: torch.tensor):
        """
        :param g: DGLGraph, possibly batched
        :param nf: Node features; expected size [#. total nodes x #. node feat dim]
        :param ef: Edge features; expected size [#. total edges x #. edge feat dim]
        :return:
        """

        # Encoder - generate transition map parameters
        # A_logit:[#. Num total nodes x #. heads]
        # b: [#. Num total nodes x #. heads]
        b, A_logit = self.enc(g, nf, ef)

        # Solve fixed point equations
        # A : [#. heads x #. total nodes x #. total nodes]
        # b : [#. heads x #. total nodes x 1]
        A, b = prepare_degree_A_b(g, A_logit, b)
        fp = self.fp_layer(A, b)  # fixed points; [#.heads x #.total nodes]

        # Decoder - generate labels
        fp = fp.transpose(1, 0)  # [#.total nodes x #.heads]
        y = self.dec(fp)  # [#. total nodes x #. sol dim]

        return y


class GraphTaskCGS(nn.Module):

    def __init__(self,
                 encoder: nn.Module,
                 readout: nn.Module,
                 decoder: nn.Module,
                 gamma: float,
                 tol: float = 1e-6,
                 max_iter: int = 50,
                 non_linear: str = None):
        super(GraphTaskCGS, self).__init__()

        self.enc = encoder
        self.fp_layer = FixedPointLayer(gamma=gamma,
                                        activation=non_linear,
                                        tol=tol,
                                        max_iter=max_iter)
        self.readout = readout
        self.dec = decoder

        # CGP options
        self.gamma = gamma

    def forward(self, g: dgl.graph, nf: torch.tensor, ef: torch.tensor):
        """
        :param g: DGLGraph, possibly batched
        :param nf: Node features; expected size [#. total nodes x #. node feat dim]
        :param ef: Edge features; expected size [#. total edges x #. edge feat dim]
        :return:
        """

        # A_logit:[#. Num total nodes x #. heads]
        # b: [#. Num total nodes x #. heads]
        b, A_logit = self.enc(g, nf, ef)

        # Solve fixed point equations
        # A : [#. heads x #. total nodes x #. total nodes]
        # b : [#. heads x #. total nodes x 1]
        A, b = prepare_degree_A_b(g, A_logit, b)
        fp = self.fp_layer(A, b)  # fixed points; [#.heads x #.total nodes]

        # Decoder - generate labels
        fp = fp.transpose(1, 0)  # [#.total nodes x #.heads]
        readout = self.readout(g, fp)
        y = self.dec(readout)
        return y
