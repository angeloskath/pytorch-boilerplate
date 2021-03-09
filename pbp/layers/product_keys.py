"""Implement the product key memory layer from 'Large Memory Layers with
Product Keys'."""

from math import sqrt

import torch


class ProductKeyMemory(torch.nn.Module):
    """Implement a large memory layer with product keys by the homonymous
    paper.

    Arguments
    ---------
        input_dims: int, the input dimensionality
        output_dims: int, the output dimensionality
        n_keys: int, the number of keys in the memory (default: 4096)
        topk: int, the number of features to consider for every call
              (default: 10)
    """
    def __init__(self, input_dims, output_dims, n_keys=4096, topk=10):
        super().__init__()

        self.values = torch.nn.Parameter(torch.randn(n_keys, output_dims)*0.01)

        n_keys_small = int(sqrt(n_keys))
        assert n_keys_small**2 == n_keys
        self.keys1 = torch.nn.Parameter(
            torch.randn(n_keys_small, input_dims//2)*0.01
        )
        self.keys2 = torch.nn.Parameter(
            torch.randn(n_keys_small, input_dims//2)*0.01
        )

        self.topk = topk

    def forward(self, x):
        *shapes, D = x.shape

        scores1_full = torch.einsum("...d,bd->...b", x[..., :D//2], self.keys1)
        scores2_full = torch.einsum("...d,bd->...b", x[..., D//2:], self.keys2)

        scores1, indices1 = torch.topk(scores1_full, self.topk, dim=-1)
        scores2, indices2 = torch.topk(scores2_full, self.topk, dim=-1)

        indices12 = (
            indices1[..., :, None] * len(self.keys1) +
            indices2[..., None, :]
        )
        scores12 = scores1[..., :, None] * scores2[..., None, :]
        scores12 = scores12.view(*(shapes + [-1,]))
        indices12 = indices12.view(*(shapes + [-1,]))

        scores, indices = torch.topk(scores12, self.topk, dim=-1)
        value_indices = indices12.view(-1, self.topk**2)
        value_indices = value_indices[
            torch.arange(len(value_indices), device=x.device)[:, None],
            indices.view(-1, self.topk)
        ].view_as(indices)

        y = torch.einsum(
            "...b,...bm->...m",
            torch.softmax(scores, dim=-1),
            self.values[value_indices]
        )

        return y
