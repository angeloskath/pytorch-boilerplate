#!/usr/bin/env python

import argparse
from functools import reduce
import operator

from einops.layers.torch import Rearrange
import torch
import torchvision

from pbp import Experiment, create_trainer
from pbp.callbacks import ModelCheckpoint, StdoutLogger, TxtLogger
from pbp.callbacks.distributed import DistributedSetup
from pbp.callbacks.wandb import WandB


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


class Datasets:
    def get_training(self, path:str, batch_size:int = 128, distributed:bool = True):
        self.data_path = path
        self.distributed = distributed

        mnist = torchvision.datasets.MNIST(
            path,
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )
        if self.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(mnist)
        else:
            sampler = None

        return torch.utils.data.DataLoader(
            dataset=mnist,
            sampler=sampler,
            batch_size=batch_size
        )

    def get_validation(self, batch_size:int = 128):
        mnist = torchvision.datasets.MNIST(
            self.data_path,
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )
        if self.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(mnist)
        else:
            sampler = None

        return torch.utils.data.DataLoader(
            dataset=mnist,
            sampler=sampler,
            batch_size=batch_size
        )


if __name__ == "__main__":
    datasets = Datasets()
    exp = Experiment(
        model=Net((1, 28, 28), 10),
        train_data=datasets.get_training,
        val_data=datasets.get_validation,
        trainer=create_trainer(training_step, validation_step),
        callbacks=[
            DistributedSetup,
            ModelCheckpoint,
            StdoutLogger,
            #WandB
        ]
    )
    exp.run()

