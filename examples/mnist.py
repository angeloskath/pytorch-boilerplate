#!/usr/bin/env python

import argparse
from functools import reduce
import operator

from einops.layers.torch import Rearrange
import torch
import torchvision

from pbp import Experiment, create_trainer


class Net(torch.nn.Module):
    def __init__(self, input_shape, output_classes):
        super().__init__()
        input_dims = reduce(operator.mul, input_shape)

        self.nn = torch.nn.Sequential(
            Rearrange("b c h w -> b (c h w)"),
            torch.nn.Linear(input_dims, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_classes)
        )

    def forward(self, x):
        return self.nn(x)

def training_step(experiment, model, batch):
    x, y = batch
    y_hat = model(x)
    loss = torch.nn.functional.cross_entropy(y_hat, y)
    print(loss.item())

    return loss

def validation_step(experiment, model, batch):
    x, y = batch
    y_hat = model(x)
    acc = (y_hat.argmax(dim=-1) == y).float().mean()
    print(acc.item())


def get_optimizer(lr:float = 0.1, momentum:float = 0.9, experiment=None):
    return torch.optim.SGD(
        experiment.model.parameters(),
        lr=lr,
        momentum=momentum
    )


if __name__ == "__main__":
    mnist = torchvision.datasets.MNIST(
        "/tmp/",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    val_mnist = torchvision.datasets.MNIST(
        "/tmp/",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    exp = Experiment(
        model=Net((1, 28, 28), 10),
        train_data=torch.utils.data.DataLoader(mnist, batch_size=256),
        val_data=torch.utils.data.DataLoader(val_mnist, batch_size=256),
        optimizer=get_optimizer,
        trainer=create_trainer(training_step, validation_step)
    )
    exp.run()
