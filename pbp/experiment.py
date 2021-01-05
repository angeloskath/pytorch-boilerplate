"""Module that provides the Experiment class which is responsible for managing
and running experiments."""

import torch

from .factory import ObjectFactory, EmptyFactory, CallableFactory


class Experiment:
    """Represents an experiment.

    Contains a key-value store that allows for objects to be available
    experiment wide as well as a global variable holding the currently running
    experiment.
    """
    _active_experiment = None

    def __init__(self, model, train_data, val_data=EmptyFactory(None),
                 optimizer=None, callbacks=None, trainer=None):
        self.model_factory = self._get_model_factory(model)
        self.train_data_factory = \
            self._get_dataset_factory( train_data, "train_data")
        self.val_data_factory = self._get_dataset_factory(val_data, "val_data")

        self._items = {}

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
        if isinstance(train_data, torch.utils.data.DataLoader):
            return EmptyFactory(data)
        #elif isinstance(train_data, torch.utils.data.Dataset):
        #    raise NotImplementedError()
        elif isinstance(train_data, ObjectFactory):
            return data
        elif callable(model):
            return CallableFactory(
                data,
                namespace=namespace
            )

        raise ValueError(("The passed {} should be an instance of "
                          "(torch.utils.data.DataLoader, "
                          "pbp.factory.ObjectFactory, callable) but it was none "
                          "of the above.").format(namespace))

    def __getitem__(self, key):
        return self._items[key]

    def __setitem__(self, key, val):
        self._items[key] = val

    @property
    def active(self):
        if self._active_experiment is None:
            raise RuntimeError(("An experiment needs to be running "
                                "for `active` to work."))
        return self._active_experiment

    def run(self):
        """Execute the experiment by collecting the arguments, creating the
        data loaders, models, callbacks, optimizers etc and then use the
        trainer to actually train."""
        self._active_experiment = self

        # Collect the arguments from all argument sources and build all the
        # components for the experiment
        arguments = self.arguments = self._collect_arguments()
        self.train_data = self.train_data_factory.from_dict(arguments)
        self.val_data = self.val_data_factory.from_dict(arguments)
        self.model = self.model_factory.from_dict(arguments)
        self.optimizer = self.optimizer_factory.from_dict(arguments)
        self.callback = self.callback_factory.from_dict(arguments)
        self.trainer = self.trainer_factory.from_dict(arguments)

        # Perform the training loop
        self.callback.on_train_start(self)
        try:
            while not self.trainer.finished:
                # One training epoch
                self.callback.on_epoch_start(self)
                for batch_idx, batch in self.train_data:
                    self.callback.on_train_batch_start(self)
                    self.trainer.train_step(batch_idx, batch)
                    self.callback.on_train_batch_stop(self)
                self.callback.on_epoch_stop(self)

                # One validation epoch if needed
                if self.trainer.validate:
                    self.callback.on_validation_start(self)
                    for batch_idx, batch in self.val_data:
                        self.callback.on_val_batch_start(self)
                        self.trainer.val_step(batch_idx, batch)
                        self.callback.on_val_batch_stop(self)
                    self.callback.on_validation_stop(self)
        finally:
            self._active_experiment = None
            self.callback.on_train_stop(self)
