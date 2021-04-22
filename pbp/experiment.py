"""Module that provides the Experiment class which is responsible for managing
and running experiments."""

import argparse
from contextlib import contextmanager
import os
from os import path

import torch
import yaml
from yaml import Loader

from .callbacks import CallbackListFactory
from .factory import ObjectFactory, EmptyFactory, CallableFactory
from .optimizers import SingleOptimizerFactory
from .trainer import Trainer


class Experiment:
    """Represents an experiment.

    Contains a key-value store that allows for objects to be available
    experiment wide as well as a global variable holding the currently running
    experiment.
    """
    _active_experiment = None

    def __init__(self, model, train_data, val_data=EmptyFactory(None),
                 optimizer=None, callbacks=[], trainer=None):
        self.model_factory = self._get_model_factory(model)
        self.train_data_factory = \
            self._get_dataset_factory( train_data, "train_data")
        self.val_data_factory = self._get_dataset_factory(val_data, "val_data")
        self.optimizer_factory = self._get_optimizer_factory(optimizer)
        self.callback_factory = CallbackListFactory(*callbacks)
        self.trainer_factory = self._get_trainer_factory(trainer)

        self._items = {}

    @staticmethod
    def active():
        if Experiment._active_experiment is None:
            raise RuntimeError(("An experiment needs to be running "
                                "for `active` to work."))
        return Experiment._active_experiment

    def _get_model_factory(self, model):
        if isinstance(model, torch.nn.Module):
            return EmptyFactory(model)
        elif isinstance(model, ObjectFactory):
            return model
        elif callable(model):
            return CallableFactory(model)

        raise ValueError(("The passed model should be an instance of "
                          "(torch.nn.Module, pbp.factory.ObjectFactory, "
                          "callable) but it was none of the above."))

    def _get_dataset_factory(self, data, namespace):
        if isinstance(data, torch.utils.data.DataLoader):
            return EmptyFactory(data)
        elif isinstance(data, ObjectFactory):
            return data
        elif callable(data):
            return CallableFactory(
                data,
                namespace=namespace
            )

        raise ValueError(("The passed {} should be an instance of "
                          "(torch.utils.data.DataLoader, "
                          "pbp.factory.ObjectFactory, callable) but it was none "
                          "of the above.").format(namespace))

    def _get_optimizer_factory(self, optimizer):
        if isinstance(optimizer, torch.optim.Optimizer):
            return EmptyFactory(optimizer)
        elif isinstance(optimizer, ObjectFactory):
            return optimizer
        elif callable(optimizer):
            return CallableFactory(optimizer)
        elif optimizer is None:
            return SingleOptimizerFactory()

        raise ValueError(("The passed optimizer should be an instance of "
                          "(torch.optim.Optimizer, pbp.factory.ObjectFactory, "
                          "callable) but it was none of the above."))

    def _get_trainer_factory(self, trainer):
        if isinstance(trainer, Trainer):
            return EmptyFactory(trainer)
        elif isinstance(trainer, ObjectFactory):
            return trainer
        elif callable(trainer):
            return CallableFactory(trainer)

        raise ValueError(("The passed trainer should be an instance of "
                          "(pbp.trainer.Trainer, pbp.factory.ObjectFactory, "
                          "callable) but it was none of the above"))

    def _get_default_output_dir(self):
        return path.join(
            os.getcwd(),
            "experiment"
        )

    def _collect_arguments(self, argv):
        # Define the parser
        parser = argparse.ArgumentParser(conflict_handler="resolve")
        parser.add_argument(
            "--config",
            help="Load the configuration from this file"
        )
        parser.add_argument(
            "--verbose",
            type=int,
            default=0,
            help=("Control the verbosity of several logging components "
                  "(default: 0 which means quiet)")
        )
        parser.add_argument(
            "--output_dir",
            default=self._get_default_output_dir(),
            help="Use this directory as the root for writing the results"
        )
        parser.add_argument(
            "--cuda",
            default=torch.cuda.is_available(),
            type=bool,
            help=("Set the device to cuda (default: "
                  "{})").format(torch.cuda.is_available())
        )
        self.train_data_factory.add_to_parser(parser)
        self.val_data_factory.add_to_parser(parser)
        self.model_factory.add_to_parser(parser)
        self.optimizer_factory.add_to_parser(parser)
        self.callback_factory.add_to_parser(parser)
        self.trainer_factory.add_to_parser(parser)

        # Parse the arguments once in order to get the config file to get
        # default values from
        args = parser.parse_args(argv)
        if args.config is not None:
            with open(args.config, "r") as f:
                config_args = yaml.load(f, Loader=Loader)
            args = argparse.Namespace(**config_args)
            args = parser.parse_args(argv, args)

        # Transform the arguments into a dictionary add the experiment and
        # return them
        args = vars(args)
        args["experiment"] = self

        return args

    def __contains__(self, key):
        return key in self._items

    def __getitem__(self, key):
        return self._items[key]

    def __setitem__(self, key, val):
        self._items[key] = val

    def _batch_to_cuda(self, batch, cuda):
        if not cuda:
            return batch

        if torch.is_tensor(batch):
            return batch.cuda()
        if isinstance(batch, (list, tuple)):
            return [self._batch_to_cuda(b, cuda) for b in batch]
        if isinstance(batch, dict):
            return {
                k: self._batch_to_cuda(v, cuda)
                for k, v in batch.items()
            }
        return batch

    @contextmanager
    def prepare(self, argv=None):
        """Create the data loaders, models, callbacks, optimizers, etc and
        yield in order for code to use the experiment. See run() for a usage
        example."""
        Experiment._active_experiment = self

        # Collect the arguments from all argument sources and build all the
        # components for the experiment
        arguments = self.arguments = self._collect_arguments(argv)
        self.train_data = self.train_data_factory.from_dict(arguments)
        self.val_data = self.val_data_factory.from_dict(arguments)
        self.model = self.model_factory.from_dict(arguments)
        self.optimizer = self.optimizer_factory.from_dict(arguments)
        self.callback = self.callback_factory.from_dict(arguments)
        self.trainer = self.trainer_factory.from_dict(arguments)

        try:
            yield None
        finally:
            Experiment._active_experiment = None

            self.arguments = None
            self.train_data = None
            self.val_data = None
            self.model = None
            self.optimizer = None
            self.callback = None
            self.trainer = None

    def run(self, argv=None):
        """Execute the experiment by collecting the arguments, creating the
        data loaders, models, callbacks, optimizers etc and then use the
        trainer to actually train."""
        with self.prepare(argv):
            # Make local variable copy
            cuda = self.arguments["cuda"]

            # Set model to the correct device
            if cuda:
                self.model.cuda()

            # Perform the training loop
            self.callback.on_train_start(self)
            try:
                while not self.trainer.finished(self):
                    self.trainer.start_epoch(self)

                    # One training epoch
                    self.callback.on_epoch_start(self)
                    for batch in self.train_data:
                        batch = self._batch_to_cuda(batch, cuda)
                        self.callback.on_train_batch_start(self)
                        self.trainer.train_step(self, batch)
                        self.callback.on_train_batch_stop(self)
                    self.callback.on_epoch_stop(self)

                    # One validation epoch if needed
                    if self.trainer.validate(self):
                        self.callback.on_validation_start(self)
                        for batch in self.val_data:
                            batch = self._batch_to_cuda(batch, cuda)
                            self.callback.on_val_batch_start(self)
                            self.trainer.val_step(self, batch)
                            self.callback.on_val_batch_stop(self)
                        self.callback.on_validation_stop(self)
            finally:
                self.callback.on_train_stop(self)

    def validate(self, argv=None):
        """Execute just a validation run."""
        with self.prepare(argv):
            # Make local variable copy
            cuda = self.arguments["cuda"]

            # Set model to the correct device
            if cuda:
                self.model.cuda()

            # Perform the training loop
            self.callback.on_train_start(self)
            try:
                self.callback.on_validation_start(self)
                for batch in self.val_data:
                    batch = self._batch_to_cuda(batch, cuda)
                    self.callback.on_val_batch_start(self)
                    self.trainer.val_step(self, batch)
                    self.callback.on_val_batch_stop(self)
                self.callback.on_validation_stop(self)
            finally:
                self.callback.on_train_stop(self)
