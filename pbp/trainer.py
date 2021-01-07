"""Implement the trainer interface required by pbp.experiment.Experiment."""

from functools import partial

import torch


class Trainer:
    def start_epoch(self, experiment):
        raise NotImplementedError()

    def finished(self, experiment):
        raise NotImplementedError()

    def validate(self, experiment):
        raise NotImplementedError()

    def train_step(self, experiment, batch_idx, batch):
        raise NotImplementedError()

    def val_step(self, experiment, batch_idx, batch):
        raise NotImplementedError()


class BaseTrainer(Trainer):
    def __init__(self, epochs: int = 1, grad_accumulate: int = 1):
        self.epochs = epochs
        self.grad_accumulate = grad_accumulate

        self.current_epoch = 0
        self.current_steps = 0

    def start_epoch(self, experiment):
        self.current_epoch += 1

    def finished(self, experiment):
        return self.current_epoch >= self.epochs

    def validate(self, experiment):
        return isinstance(experiment.val_data, torch.utils.data.DataLoader)

    def train_step(self, experiment, batch_idx, batch):
        model = experiment.model
        optimizer = experiment.optimizer

        # Put the model in training mode and zero the gradients if needed
        model.train()
        if (self.current_steps % self.grad_accumulate) == 0:
            optimizer.zero_grad()

        # Compute the loss
        loss = self._compute_loss(experiment, model, batch)

        # Increase the number of steps and perform a gradient update if needed
        self.current_steps += 1
        if (self.current_steps % self.grad_accumulate) == 0:
            optimizer.step()

    def val_step(self, experiment, batch):
        model = experiment.model

        # Put the model in evaluation mode and 
        model.eval()

        # Compute and log the validation metrics
        self._compute_validation(experiment, model, batch)

    def _compute_loss(self, experiment, model, batch):
        pass

    def _compute_validation(self, experiment, model, batch):
        pass


class FunctionTrainer(BaseTrainer):
    def __init__(self, train_step, val_step, epochs: int = 1,
                 grad_accumulate: int = 1):
        super().__init__(epochs, grad_accumulate)

        self._train_step = train_step
        self._val_step = val_step

    def _compute_loss(self, experiment, model, batch):
        return self._train_step(experiment, model, batch)

    def _compute_validation(self, experiment, model, batch):
        return self._val_step(experiment, model, batch)


def create_trainer(train_step, val_step=lambda *args: None):
    return partial(FunctionTrainer, train_step, val_step)
