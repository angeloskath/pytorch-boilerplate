#!/usr/bin/env python

import torch
import torchvision

from fast_transformers.builders import TransformerEncoderBuilder

from pbp import Experiment, create_trainer
from pbp.callbacks import StdoutLogger
from pbp.callbacks.lr_schedule import WarmupLR, MultiplicativeLRSchedule
from pbp.layers.perceiver import Perceiver
from pbp.layers.positional_encoding import FixedPositionalEncoding

class MNISTPerceiver(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.pe = FixedPositionalEncoding()
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=4,
            n_heads=3,
            query_dimensions=32,
            value_dimensions=32,
            attention_type="full"
        )
        self.perceiver = Perceiver(
            32*3,
            [builder.get() for i in range(2)],
            n_latent=32,
            latent_dims=32*3,
            attention_heads=3
        )
        self.fc = torch.nn.Linear(32*3, 10)

    def forward(self, images):
        B = len(images)
        x = torch.linspace(-1, 1, images.shape[3], device=images.device)
        y = torch.linspace(-1, 1, images.shape[2], device=images.device)
        x = x[None, None, None, :].repeat(*(images.shape[:-1] + (1,)))
        y = y[None, None, :, None].repeat(*(images.shape[:-2] + (1,) + images.shape[-1:]))
        images = torch.stack([x, y, images*2 - 1], dim=-1)
        images = self.pe(images).view(B, -1, 3*32)
        images = self.perceiver(images)
        return self.fc(images.mean(1))


class MNISTTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.pe = FixedPositionalEncoding()
        self.transformer = TransformerEncoderBuilder.from_kwargs(
            n_layers=8,
            n_heads=3,
            query_dimensions=32,
            value_dimensions=32,
            attention_type="full"
        ).get()
        self.class_embedding = torch.nn.Parameter(torch.randn(1, 1, 96))
        self.fc = torch.nn.Linear(32*3, 10)

    def forward(self, images):
        B = len(images)
        x = torch.linspace(-1, 1, images.shape[3], device=images.device)
        y = torch.linspace(-1, 1, images.shape[2], device=images.device)
        x = x[None, None, None, :].repeat(*(images.shape[:-1] + (1,)))
        y = y[None, None, :, None].repeat(*(images.shape[:-2] + (1,) + images.shape[-1:]))
        images = torch.stack([x, y, images*2 - 1], dim=-1)
        images = self.pe(images).view(B, -1, 3*32)
        ce = self.class_embedding.repeat(B, 1, 1)
        images = torch.cat([ce, images], dim=1)
        images = self.transformer(images)
        return self.fc(images[:, 0])


def get_model(model_type:str = "perceiver"):
    if model_type == "perceiver":
        return MNISTPerceiver()
    elif model_type == "transformer":
        return MNISTTransformer()
    else:
        raise NotImplementedError()


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


def make_dataloader(dset):
    def inner(batch_size:int = 256):
        return torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=True
        )
    return inner


if __name__ == "__main__":
    mnist = torchvision.datasets.FashionMNIST(
        "/tmp/",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    val_mnist = torchvision.datasets.FashionMNIST(
        "/tmp/",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    exp = Experiment(
        model=get_model,
        train_data=make_dataloader(mnist),
        val_data=make_dataloader(val_mnist),
        trainer=create_trainer(training_step, validation_step),
        callbacks=[
            WarmupLR,
            MultiplicativeLRSchedule,
            StdoutLogger,
        ]
    )
    exp.run()
