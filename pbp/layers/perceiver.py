"""Implement the perceiver architecture from 'Perceiver: General Perception
with Iterative Attention'."""

from functools import partial

import torch

from fast_transformers.attention import AttentionLayer
from fast_transformers.builders import AttentionBuilder
from fast_transformers.masking import FullMask, LengthMask


class Perceiver(torch.nn.Module):
    """See https://arxiv.org/abs/2103.03206 .

    Arguments
    ---------
        input_dims: int, the input feature size
        n_latent: int, the size of the latent sequence
        latent_dims: int, the size of the latent vectors
        latent_transformers: list[TransformerEncoder], a list of transformers
                             to change the latent sequence
        attention_heads: int, number of heads for the attention to the input
        attention_query_dimensions: int, the size of the queries and keys for
                                    the attention
        attention_type: str, the attention to use for the input
        attention_parameters: dict, the arguments for the attention to the
                              input
        share_attention: bool, whether to share the attention layer weights
    """
    def __init__(self, input_dims, latent_transformers, n_latent=512,
                 latent_dims=512, attention_heads=8,
                 attention_query_dimensions=None, attention_value_dimensions=None,
                 attention_type="full", attention_parameters={},
                 share_attention=False):
        super().__init__()

        self.latent_sequence = torch.nn.Parameter(
            torch.randn(n_latent, latent_dims)
        )
        self.latent_transformers = torch.nn.ModuleList(latent_transformers)

        attention_builder = AttentionBuilder.from_dictionary(
            attention_parameters
        )
        attention_layer = partial(
            AttentionLayer,
            d_model=input_dims,
            n_heads=attention_heads,
            d_keys=attention_query_dimensions,
            d_values=attention_value_dimensions
        )
        N = len(self.latent_transformers)
        if share_attention:
            self.cross_attention = [
                attention_layer(attention_builder.get(attention_type))
            ]*N
        else:
            self.cross_attention = [
                attention_layer(attention_builder.get(attention_type))
                for _ in range(N)
            ]
        self.cross_attention = torch.nn.ModuleList(self.cross_attention)

    def forward(self, x):
        # Extract shapes into local variables
        B, M, _ = x.shape
        N = self.latent_sequence.shape[0]

        # Local reference to the latent sequence
        y = self.latent_sequence
        y = y[None].expand(B, -1, -1)

        # Create the cross attention full masks
        attn_mask = FullMask(N=N, M=M, device=x.device)
        query_lengths = LengthMask(x.new_full((B,), N, dtype=torch.int64))
        key_lengths = LengthMask(x.new_full((B,), M, dtype=torch.int64))

        # Transform the latent sequence
        blocks = zip(self.cross_attention, self.latent_transformers)
        for cross_attention, latent_transformer in blocks:
            y = cross_attention(y, x, x, attn_mask, query_lengths, key_lengths)
            y = latent_transformer(y)

        # Return the transformed sequence in the end
        return y
