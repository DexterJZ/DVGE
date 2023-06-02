"""ops.py"""

import torch
import torch.nn.functional as F


def recon_loss(x, x_recon):
    n = x.size(0)
    loss = F.binary_cross_entropy_with_logits(x_recon, x,
                                              size_average=False).div(n)
    return loss


def cls_loss(a_sens, b_logits):
    n = a_sens.size(0)
    loss = F.binary_cross_entropy_with_logits(b_logits, a_sens,
                                              size_average=False).div(n)
    return loss


def kl_divergence(mu, logvar):
    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
    return kld


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)


def permute_sens_dims(z, n_sens):
    assert z.dim() == 2

    B, D = z.size()
    sens_dims = range(D - n_sens, D)
    perm_z = []
    count = 0

    for z_j in z.split(1, 1):
        if count in sens_dims:
            perm = torch.randperm(B).to(z.device)
            perm_z_j = z_j[perm]
            perm_z.append(perm_z_j)
        else:
            perm_z.append(z_j)

        count += 1

    return torch.cat(perm_z, 1)
