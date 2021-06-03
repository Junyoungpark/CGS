import torch
import torch.nn as nn


class FixedPointLayer(nn.Module):
    def __init__(self,
                 gamma: float,
                 activation: str,
                 tol: float = 1e-6,
                 max_iter: int = 50,
                 alpha: float = 1.0):

        super(FixedPointLayer, self).__init__()
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha
        self.act = getattr(nn, activation)()
        self._act_str = activation

        self.frd_itr = None  # forward iterations
        self.bwd_itr = None  # backward iterations
        self.A_max = None

    def forward(self, A, b):
        """
        :param A: The entities of A matrix; Expected size [#.heads x Num edges x Num edges]
        :param b: The entities of B matrix; Expected size [#.heads x Num nodes x 1]
        :return: z: Fixed points of the input linear systems; size [#. heads x Num nodes]
        """

        z, self.frd_itr = self.solve_fp_eq(A, b,
                                           self.gamma,
                                           self.act,
                                           self.max_iter,
                                           self.tol)
        self.A_max = A.max()

        # re-engage autograd and add the gradient hook

        z = self.act(self.gamma * torch.bmm(A, z) + b)  # [#.heads x #.Nodes]

        if z.requires_grad:
            y0 = (self.gamma * torch.bmm(A, z) + b).detach().requires_grad_()
            z_next = self.act(y0)
            z_next.sum().backward()
            dphi = y0.grad
            J = self.gamma * (dphi * A).transpose(2, 1)

            def modify_grad(grad):
                y, bwd_itr = self.solve_fp_eq(J,
                                              grad,
                                              1.0,
                                              nn.Identity(),
                                              self.max_iter,
                                              self.tol)

                return y

            z.register_hook(modify_grad)
        z = z.squeeze(dim=-1)  # drop dummy dimension
        return z

    @staticmethod
    @torch.no_grad()
    def solve_fp_eq(A, b,
                    gamma: float,
                    act: nn.Module,
                    max_itr: int,
                    tol: float):
        """
        Find the fixed point of x = gamma * A * x + b
        """

        # initialize the solution w/ b is done to support backward scheme also.
        # o/w Pytorch complains like 'backward not multiplied by grad_output'

        x = torch.zeros_like(b, device=b.device)  # [#. heads x #.total nodes - possibly batched]
        itr = 0
        while itr < max_itr:
            x_next = act(gamma * torch.bmm(A, x) + b)
            # x_next = self.alpha * x_next + (1.0 - self.alpha) * x  # damping
            g = x - x_next
            if torch.norm(g) < tol:
                break
            x = x_next
            itr += 1
        return x, itr
