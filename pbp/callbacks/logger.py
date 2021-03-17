"""Logger callback logs scalars to different files so that they can be
processed later."""

from collections import defaultdict
import os
from os import path
import sys

from ..utils import AverageMeter
from .base import Callback


class LoggerList:
    def __init__(self, logger_a, logger_b):
        self.logger_a = logger_a
        self.logger_b = logger_b

    def log(self, key, value):
        self.logger_a.log(key, value)
        self.logger_b.log(key, value)


class Logger(Callback):
    def log(self, key, value):
        raise NotImplementedError()

    def on_train_start(self, experiment):
        if "logger" in experiment:
            experiment["logger"] = LoggerList(experiment["logger"], self)
        else:
            experiment["logger"] = self


class TxtLogger(Logger):
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

    def on_train_stop(self, experiment):
        for k, f in self._files.items():
            f.close()
        self._files.clear()
            
    def on_train_batch_stop(self, experiment):
        self._write_values(experiment)

    def on_val_batch_stop(self, experiment):
        self._write_values(experiment)


class StdoutLogger(Logger):
    """Log scalars to stdout."""
    def __init__(self):
        self._values = defaultdict(AverageMeter)

    def log(self, key, value):
        self._values[key] += value

    def _write_values(self, experiment):
        msg = " - ".join([
            "{}: {}".format(k, v.average_value)
            for k, v in self._values.items()
        ])

        if sys.stdout.isatty():
            msg += "\b"*len(msg)
        else:
            msg += "\n"

        print(msg, flush=True, end="")

    def _clear_values(self):
        if sys.stdout.isatty() and len(self._values) > 0:
            print()
        self._values.clear()

    def on_epoch_start(self, experiment):
        self._clear_values()

    def on_train_batch_stop(self, experiment):
        self._write_values(experiment)

    def on_validation_start(self, experiment):
        self._clear_values()

    def on_val_batch_stop(self, experiment):
        self._write_values(experiment)

    def on_train_stop(self, experiment):
        self._clear_values()
