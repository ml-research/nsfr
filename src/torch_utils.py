import torch
import numpy as np


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.
    from https://github.com/pytorch/pytorch/issues/2591#issuecomment-364474328
    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def weight_sum(W_l, H):
    # W: C
    # H: C * B * G
    W_ex = W_l.unsqueeze(dim=-1).unsqueeze(dim=-1).expand_as(H)
    # C * B * G
    WH = W_ex * H
    # B * G
    WH_sum = torch.sum(WH, dim=0)
    return WH_sum


def softor(xs, dim=0, gamma=0.01):
    """The softor function.

    Args:
        xs (tensor or list(tensor)): The input tensor.
        dim (int): The dimension to be removed.
        gamma (float: The smooth parameter for logsumexp. 
    Returns:
        log_sum_exp (tensor): The result of taking or along dim.
    """
    # xs is List[Tensor] or Tensor
    if not torch.is_tensor(xs):
        xs = torch.stack(xs, dim)
    log_sum_exp = gamma*logsumexp(xs * (1/gamma), dim=dim)
    if log_sum_exp.max() > 1.0:
        return log_sum_exp / log_sum_exp.max()
    else:
        return log_sum_exp


def print_valuation(valuation, atoms, n=40):
    """Print the valuation tensor.

    Print the valuation tensor using given atoms.
    Args:
        valuation (tensor;(B*G)): A valuation tensor.
        atoms (list(atom)): The ground atoms.
    """
    for b in range(valuation.size(0)):
        print('===== BATCH: ', b, '=====')
        v = valuation[b].detach().cpu().numpy()
        idxs = np.argsort(-v)
        for i in idxs:
            if v[i] > 0.1:
                print(i, atoms[i], ': ', round(v[i], 3))
