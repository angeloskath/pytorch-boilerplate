#!/usr/bin/env python

import argparse
from functools import reduce
import operator

from einops.layers.torch import Rearrange
import torch
import torchvision

from pbp import Experiment, create_trainer
from pbp.callbacks import ModelCheckpoint, StdoutLogger, TxtLogger


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
    Experiment.active()["logger"].log("loss", loss.item())

    return loss


def validation_step(experiment, model, batch):
    x, y = batch
    y_hat = model(x)
    acc = (y_hat.argmax(dim=-1) == y).float().mean()
    experiment["logger"].log("val_acc", acc.item())


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
        trainer=create_trainer(training_step, validation_step),
        callbacks=[
            ModelCheckpoint.factory,
            StdoutLogger(),
            TxtLogger()
        ]
    )
    exp.run()
