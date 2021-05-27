"""Weights & biases logger."""

from collections import defaultdict

try:
    import wandb
except ImportError:
    pass


from ..utils import AverageMeter, rank_zero_method
from .logger import Logger


class WandB(Logger):
    """Log the metrics in weights and biases.

    Arguments
    ---------
        wandb_project: str, the project name to use in weights and biases
                       (default: '')
        wandb_watch: bool, use wandb.watch() on the model (default: True)
        wandb_per_epoch: bool, log the metrics per epoch or per update
                         (default: True)
        wandb_log_frequency: int, log every that many batches (default: 10)
        wandb_watch_log_frequency: int, the log frequency passed to wandb.watch
                                   (default: 1000)
        wandb_run_id: str, the run id in order to resume a previously preempted
                      run (default: '')
    """
    def __init__(self, wandb_project:str = "", wandb_watch:bool = True,
                 wandb_per_epoch:bool = True, wandb_log_frequency:int = 10,
                 wandb_watch_log_frequency:int = 1000, wandb_run_id:str = ""):
        self.project = wandb_project
        self.watch = wandb_watch
        self.log_frequency = wandb_log_frequency
        self.watch_log_frequency = wandb_watch_log_frequency
        self.per_epoch = wandb_per_epoch
        self.run_id = wandb_run_id
        self._values = defaultdict(AverageMeter)
        self._validation_batches = 0

    @rank_zero_method
    def on_train_start(self, experiment):
        super().on_train_start(experiment)

        # Login to wandb
        wandb.login()

        # Init the run
        wandb.init(
            project=(self.project or None),
            id=(self.run_id or None),
            config=dict(experiment.arguments.items()),
            resume="allow"
        )

        if self.watch:
            wandb.watch(experiment.model, log_freq=self.watch_log_frequency)

    def log(self, key, value):
        self._values[key] += value

    @rank_zero_method
    def on_train_batch_stop(self, experiment):
        if self.per_epoch:
            return

        if experiment.trainer.current_steps % self.log_frequency != 0:
            return

        self._values["step"] += experiment.trainer.current_steps
        wandb.log({k: v.average_value for (k, v) in self._values.items()})
        self._values.clear()

    @rank_zero_method
    def on_epoch_start(self, experiment):
        if len(self._values) == 0:
            return

        self._values["epoch"] += experiment.trainer.current_epoch-1
        wandb.log({k: v.average_value for (k, v) in self._values.items()})
        self._values.clear()

    @rank_zero_method
    def on_train_stop(self, experiment):
        if len(self._values) == 0:
            return

        self._values["epoch"] += experiment.trainer.current_epoch
        wandb.log({k: v.average_value for (k, v) in self._values.items()})
        self._values.clear()
