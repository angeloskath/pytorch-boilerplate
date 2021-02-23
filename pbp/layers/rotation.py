"""Implement layers that predict a rotation."""

import torch


class UnitQuaternion(torch.nn.Module):
    """Predict an L2 normalized 4D vector.

    The real part of the quaternion is assumed to be the first dimension.

    Arguments
    ---------
        input_dims: int, the dimensionality of the input feature
        standardize: bool, if True ensures that the real part of the quaternion
                     is positive (default: True)
    """
    def __init__(self, input_dims, standardize=True):
        super().__init__()

        self.standardize = standardize
        self.linear = torch.nn.Linear(input_dims, 4)

    def forward(self, x):
        q = torch.nn.functional.normalize(self.linear(x), dim=-1)

        if self.standardize:
            m = (q[..., :1] > 0).float()
            q = m * q - (1-m) * q

        return q


class SVDO(torch.nn.Module):
    """Perform SVD orthogonalization introduced in [1].

    [1]: Levinson, J., Esteves, C., Chen, K., Snavely, N., Kanazawa, A.,
         Rostamizadeh, A. and Makadia, A., 2020. An analysis of svd for deep
         rotation estimation. NeurIPS 2020

    Arguments
    ---------
        input_dims: int, the dimensionality of the input feature
        enforce_rotation: bool, if set to True ensures that the determinant of
                          the predicted matrix will be 1 thus a valid rotation
                          and not a roto-reflection (default: True)
    """
    def __init__(self, input_dims, enforce_rotation=True):
        super().__init__()

        self.enforce_rotation = enforce_rotation
        self.linear = torch.nn.Linear(input_dims, 9)

    def forward(self, x):
        shape = x.shape[:-1]
        M = self.linear(x).view(*(shape + (3, 3)))

        U, S, V = torch.svd(M)

        if self.enforce_rotation:
            S_hat = torch.ones_like(S)
            S_hat[..., -1] = torch.det(
                torch.einsum("...ik,...jk->...ij", U, V)
            )
            U = U * S_hat[..., None, :]

        return torch.einsum("...ik,...jk->...ij", U, V)
