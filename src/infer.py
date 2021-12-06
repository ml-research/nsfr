
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import softor, weight_sum


class InferModule(nn.Module):
    """
    A class of differentiable foward-chaining inference.
    """

    def __init__(self, I, m, infer_step, gamma=0.01, device=None, train=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(InferModule, self).__init__()
        self.I = I
        self.infer_step = infer_step
        self.m = m
        self.C = self.I.size(0)
        self.G = self.I.size(1)
        self.gamma = gamma
        self.device = device
        self.train_ = train
        if not train:
            self.W = self.init_identity_weights(device)
        else:
            # to learng the clause weights, initialize W as follows:
            self.W = nn.Parameter(torch.Tensor(
                np.random.normal(size=(m, I.size(0)))).to(device))
        # clause functions
        self.cs = [ClauseFunction(i, I, gamma=gamma)
                   for i in range(self.I.size(0))]

        assert m == self.C, "Invalid m and C: " + \
            str(m) + ' and ' + str(self.C)

    def init_identity_weights(self, device):
        ones = torch.ones((self.C, ), dtype=torch.float32) * 100
        return torch.diag(ones).to(device)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        R = x
        for t in range(self.infer_step):
            R = softor([R, self.r(R)], dim=1, gamma=self.gamma)
        return R

    def r(self, x):
        B = x.size(0)  # batch size
        # apply each clause c_i and stack to a tensor C
        # C * B * G
        C = torch.stack([self.cs[i](x)
                        for i in range(self.I.size(0))], 0)

        # taking weighted sum using m weights and stack to a tensor H
        # m * C
        W_star = torch.softmax(self.W, 1)
        # m * C * B * G
        W_tild = W_star.unsqueeze(
            dim=-1).unsqueeze(dim=-1).expand(self.m, self.C, B, self.G)
        # m * C * B * G
        C_tild = C.unsqueeze(dim=0).expand(self.m, self.C, B, self.G)
        # m * B * G
        H = torch.sum(W_tild * C_tild, dim=1)
        # taking soft or to compose a logic program with m clauses
        # B * G
        R = softor(H, dim=0, gamma=self.gamma)
        return R


class ClauseFunction(nn.Module):
    """
    A class of the clause function.
    """

    def __init__(self, i, I, gamma=0.01):
        super(ClauseFunction, self).__init__()
        self.i = i  # clause index
        self.I = I  # index tensor C * S * G, S is the number of possible substituions
        self.L = I.size(-1)  # number of body atoms
        self.S = I.size(-2)  # max number of possible substitutions
        self.gamma = gamma

    def forward(self, x):
        batch_size = x.size(0)  # batch size
        # B * G
        V = x
        # G * S * b
        I_i = self.I[self.i, :, :, :]

        # B * G -> B * G * S * L
        V_tild = V.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.S, self.L)
        # G * S * L -> B * G * S * L
        I_i_tild = I_i.repeat(batch_size, 1, 1, 1)

        # B * G
        C = softor(torch.prod(torch.gather(V_tild, 1, I_i_tild), 3),
                   dim=2, gamma=self.gamma)
        return C
