"""Implement the simplest invertible neural network Real NVP."""

import torch


class RealNVP(torch.nn.Module):
    """RealNVP is an invertible neural network.

    See https://arxiv.org/abs/1605.08803 for details.

    Arguments
    ---------
        scale_networks: A list of modules to predict the scaling factors
        translation_networks: A list of modules to predict the translation
        masks: A list of tensors defining the parts of the input used in the
               prediction
    """
    def __init__(self, *, scale_networks, translation_networks, masks):
        super().__init__()

        self.scale_networks = torch.nn.ModuleList(scale_networks)
        self.translation_networks = torch.nn.ModuleList(translation_networks)
        self.register_buffer(
            "masks",
            torch.stack(list(masks), dim=0)
        )

    def forward(self, z):
        nets = zip(
            self.scale_networks,
            self.translation_networks,
            self.masks
        )
        x = z
        logdetj = 0
        for ns, nt, m in nets:
            x_hat = x * m
            s = ns(x_hat)
            t = nt(x_hat)
            x = x_hat + (1 - m) * (x * torch.exp(s) + t)
            logdetj = logdetj + ((1 - m) * s).sum(-1)

        return x, logdetj

    def inverse(self, x):
        nets = zip(
            self.scale_networks,
            self.translation_networks,
            self.masks
        )
        z = x
        logdetj = 0
        for ns, nt, m in reversed(list(nets)):
            z_hat = z * m
            s = ns(z_hat)
            t = nt(z_hat)
            z = z_hat + (1 - m) * (z - t) * torch.exp(-s)
            logdetj = logdetj - ((1 - m) * s).sum(-1)

        return z, logdetj
