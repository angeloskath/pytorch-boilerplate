"""Weights & biases logger."""

from collections import defaultdict

try:
    import wandb
except ImportError:
    pass


from ..utils import AverageMeter
from .logger import Logger


class WandB(Logger):
    """Log the metrics in weights and biases.

    Arguments
    ---------
        project: str, the project name to use in weights and biases
                 (default: '')
        watch: bool, use wandb.watch() on the model (default: True)
        log_frequency: int, the log frequency passed to wandb.watch
                       (default: 10)
    """
    def __init__(self, wandb_project:str = "", wandb_watch:bool = True,
                 wabdb_per_epoch:bool = True, wandb_log_frequency:int = 10):
        self.project = wandb_project
        self.watch = wandb_watch
        self.log_frequency = wandb_log_frequency
        self.per_epoch = wandb_per_epoch
        self._values = defaultdict(AverageMeter)
        self._validation_batches = 0

    def on_train_start(self, experiment):
        super().on_train_start(experiment)

        # Login to wandb
        wandb.login()

        # Init the run
        wandb.init(
            project=(self.project or None),
            config=dict(experiment.arguments.items())
        )

        if self.watch:
            wandb.watch(experiment.model, log_freq=self.log_frequency)

    def log(self, key, value):
        self._values[key] += value

    def on_train_batch_stop(self, experiment):
        if self.per_epoch:
            return

        self._values["step"] += experiment.trainer.current_steps
        wandb.log({k: v.current_value for (k, v) in self._values.items()})
        self._values.clear()

    def on_val_batch_stop(self, experiment):
        if self.per_epoch:
            return

        self._values["val_step"] += self._validation_batches
        wandb.log({k: v.current_value for (k, v) in self._values.items()})
        self._values.clear()
        self._validation_batches += 1

    def on_epoch_start(self, experiment):
        if not self.per_epoch:
            return

        if len(self._values) == 0:
            return

        self._values["epoch"] += experiment.trainer.current_epoch-1
        wandb.log({k: v.average_value for (k, v) in self._values.items()})
        self._values.clear()
