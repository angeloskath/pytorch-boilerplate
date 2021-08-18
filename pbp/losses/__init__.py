"""Implementations of different loss functions or loss utilities in PyTorch."""

import torch


def reinforce(loss, logits, baseline):
    """Return the simple REINFORCE loss with a baseline.

    NOTE: This does not add the loss tensor to the returned loss so make sure
          to add it separately!

    Arguments
    ---------
        loss: The tensor that contains the per-sample loss
        logits: The tensor that contains the per-sample logits
        baseline: The tensor that contains the baseline loss used to reduce the
                  variance of the gradient estimator
    """
    B = loss.shape[0]
    assert loss.shape == (B,) or loss.shape == (B, 1)
    assert baseline.numel() == 1
    assert logits.shape[0] == B

    return (
        (loss.view(-1, 1) - baseline.view(-1)).detach() * logits.view(B, -1)
    ).mean(1)


class Reinforce(torch.nn.Module):
    """Compute the simple REINFORCE loss with a exponential moving average
    baseline.

    Arguments
    ---------
        alpha: float, the parameter for the exponential moving
               average (default: 0.9)
    """
    def __init__(self, alpha=0.9):
        super().__init__()

        self.alpha = alpha
        self.baseline = None

    def forward(self, loss, logits):
        if self.baseline is None:
            self.baseline = torch.zeros(1, device=loss.device)
            with torch.no_grad():
                self.baseline[...] = loss.mean()

        rloss = reinforce(loss, logits, self.baseline)
        with torch.no_grad():
            self.baseline[...] = (
                self.alpha * self.baseline +
                (1-self.alpha)*loss.mean()
            )

        return rloss
