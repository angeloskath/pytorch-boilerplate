"""Logger callback logs scalars to different files so that they can be
processed later."""

import os
from os import path

from .base import Callback


class TxtLogger(Callback):
    """Logger stores scalars to files for later processing."""
    def __init__(self):
        self._files = {}
        self._values = {}

    def _get_file(self, experiment, key):
        if key not in self._files:
            log_file_path = path.join(
                experiment.arguments["output_dir"],
                "logs",
                key
            )
            if not path.exists(path.dirname(log_file_path)):
                os.makedirs(path.dirname(log_file_path))
            self._files[key] = open(log_file_path, "a")
        return self._files[key]

    def log(self, key, value):
        # TODO: Maybe treat values differently if they are tensors
        self._values[key] = value

    def _write_values(self, experiment):
        for k, v in self._values.items():
            print(
                v,
                file=self._get_file(experiment, k),
                flush=True
            )
        self._values.clear()

    def on_train_start(self, experiment):
        experiment["logger"] = self

    def on_train_stop(self, experiment):
        for k, f in self._files.items():
            f.close()
        self._files.clear()
            
    def on_train_batch_stop(self, experiment):
        self._write_values(experiment)

    def on_val_batch_stop(self, experiment):
        self._write_values(experiment)
