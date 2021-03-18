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
        out_dims: int, the dimensionality of the rotation matrix (default: 3)
    """
    def __init__(self, input_dims, enforce_rotation=True, out_dims=3):
        super().__init__()

        self.enforce_rotation = enforce_rotation
        self.out_dims = out_dims
        self.linear = torch.nn.Linear(input_dims, out_dims*out_dims)

    def forward(self, x):
        shape = x.shape[:-1]
        M = self.linear(x).view(*(shape + (self.out_dims, self.out_dims)))

        U, S, V = torch.svd(M)

        if self.enforce_rotation:
            S_hat = torch.ones_like(S)
            S_hat[..., -1] = torch.det(
                torch.einsum("...ik,...jk->...ij", U, V)
            )
            U = U * S_hat[..., None, :]

        return torch.einsum("...ik,...jk->...ij", U, V)


class GS3(torch.nn.Module):
    """Perform Gram-Schmidt orthogonalization for 3x3 matrices, namely the 6D
    representation from [1].

    [1]: Zhou, Y., Barnes, C., Lu, J., Yang, J. and Li, H., 2019. On the
         continuity of rotation representations in neural networks. CVPR 2019.

    Arguments
    ---------
        input_dims: int, the dimensionality of the input feature
    """
    def __init__(self, input_dims):
        super().__init__()
        self.linear = torch.nn.Linear(input_dims, 6)

    def forward(self, x):
        u = self.linear(x)
        u1 = u[..., :3]
        u2 = u[..., 3:]

        u1_norm = (u1*u1).sum(-1)
        u2 = u2 - ((u1*u2).sum(-1) / u1_norm)[..., None] * u1
        u2_norm = (u2*u2).sum(-1)

        e1 = u1 / torch.sqrt(u1_norm)[..., None]
        e2 = u2 / torch.sqrt(u2_norm)[..., None]
        e3 = torch.cross(e1, e2, dim=-1)

        return torch.stack([e1, e2, e3], dim=-2)
