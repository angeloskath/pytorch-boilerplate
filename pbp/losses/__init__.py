"""Implementations of different loss functions or loss utilities in PyTorch."""


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
    assert len(logits.shape) == 2 and logits.shape[0] == B

    return (
        (loss.view(-1, 1) - baseline.view(-1)).detach() * logits
    ).mean()
