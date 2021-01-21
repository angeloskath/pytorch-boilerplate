"""Implement the trainer interface required by pbp.experiment.Experiment."""

from functools import partial

import torch


class Trainer:
    @property
    def current_epoch(self):
        raise NotImplementedError()

    def start_epoch(self, experiment):
        raise NotImplementedError()

    def set_epoch(self, experiment, epoch):
        raise NotImplementedError()

    def finished(self, experiment):
        raise NotImplementedError()

    def validate(self, experiment):
        raise NotImplementedError()

    def train_step(self, experiment, batch):
        raise NotImplementedError()

    def val_step(self, experiment, batch):
        raise NotImplementedError()


class BaseTrainer(Trainer):
    def __init__(self, epochs: int = 1, grad_accumulate: int = 1):
        self.epochs = epochs
        self.grad_accumulate = grad_accumulate

        self._current_epoch = 0
        self._current_steps = 0

    @property
    def current_epoch(self):
        return self._current_epoch

    def start_epoch(self, experiment):
        self._current_epoch += 1

    def set_epoch(self, experiment, epoch):
        self._current_epoch = epoch

    @property
    def current_steps(self):
        return self._current_steps

    def set_steps(self, experiment, steps):
        self._current_steps = steps

    def finished(self, experiment):
        return self._current_epoch >= self.epochs

    def validate(self, experiment):
        return isinstance(experiment.val_data, torch.utils.data.DataLoader)

    def train_step(self, experiment, batch):
        model = experiment.model
        optimizer = experiment.optimizer

        # Put the model in training mode and zero the gradients if needed
        if not model.training:
            model.train()
        if (self._current_steps % self.grad_accumulate) == 0:
            optimizer.zero_grad()

        # Compute the loss and the gradients
        loss = self._compute_loss(experiment, model, batch)
        loss.backward()

        # Increase the number of steps and perform a gradient update if needed
        self._current_steps += 1
        if (self._current_steps % self.grad_accumulate) == 0:
            optimizer.step()

    def val_step(self, experiment, batch):
        model = experiment.model

        # Put the model in evaluation mode and 
        if model.training:
            model.eval()

        # Compute and log the validation metrics
        with torch.no_grad():
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
