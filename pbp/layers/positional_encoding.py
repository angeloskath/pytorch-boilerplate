"""Implement positional encoding layers."""

from math import log, tau as two_pi, sqrt

import torch


class FixedPositionalEncoding(torch.nn.Module):
    """A general fixed positional encoding that can be used for both the
    integer positions and real coordinates.

    The encoding is simply the concatenation of

        cos(2π σ_i x), sin(2π σ_i x)

    where x is either normalized integer coordinates or unnormalized real
    coordinates and the σ_i are frequency bands that could scale
    logarithmically or linearly.

    Arguments
    ---------
        normalize_with: float, normalize the inputs with this number (default: 1)
        n_frequencies: int, how many frequencies to use (default: 16)
        sigma_0: float, the starting frequency to use (default: 0.1)
        sigma_n: float, the final frequency to use (default: 100)
        frequency_scaling: {log, linear}, choose how to scale the frequencies
                           (default: log)
    """
    def __init__(self, normalize_with=1., n_frequencies=16, sigma_0=0.1,
                 sigma_n=100, frequency_scaling="log"):
        super().__init__()

        self.register_buffer("normalize_with", torch.tensor(1/normalize_with))
        self.register_buffer(
            "sigmas",
            torch.exp(
                torch.linspace(log(sigma_0), log(sigma_n), n_frequencies)
            )
        )

    def forward(self, x):
        # Cast to float and normalize
        x = x.float() * self.normalize_with

        # Embed
        cosx = torch.cos(two_pi * self.sigmas * x[..., None])
        sinx = torch.sin(two_pi * self.sigmas * x[..., None])
        pe_x = torch.cat([cosx, sinx], dim=-1)

        # Flatten the last two dimensions
        shape = pe_x.shape

        return pe_x.view(*(shape[:-2] + (-1,)))


class RFF(torch.nn.Module):
    """RFF implements Random Fourier Features for the gaussian kernel to be
    used as a positional encoding since the dot product will be the value of
    the gaussian kernel in expectation.

    This was introduced as a positional encoding in [1].

    [1]: Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S.,
         Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J.T. and Ng, R.,
         2020.  Fourier features let networks learn high frequency functions in
         low dimensional domains. NeurIPS 2020

    Arguments
    ---------
        input_dims: int, the dimensionality of the input feature
        feature_dims: int, the dimensionality of the random Fourier features
        sigma: float, the standard deviation from which to draw the random
               matrix (defines the gamma parameter for the gaussian kernel)
               (default: 1.)
    """
    def __init__(self, input_dims, feature_dims, sigma=1.0):
        super().__init__()

        self.register_buffer(
            "beta",
            torch.randn(feature_dims//2, input_dims) * sigma
        )

    def forward(self, x):
        bx = torch.einsum("...i,ji->...j", x, self.beta)
        scale = sqrt(1 / len(self.beta))
        return torch.cat([torch.cos(bx), torch.sin(bx)], dim=-1) * scale
