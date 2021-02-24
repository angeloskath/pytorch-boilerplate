"""Weights & biases logger."""

from collections import defaultdict

try:
    import wandb
except ImportError:
    pass


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
    def __init__(self, project:str = "", watch:bool = True, log_frequency:int = 10):
        self.project = project
        self.watch = watch
        self.log_frequency = log_frequency
        self._values = {}
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
        self._values[key] = value

    def on_train_batch_stop(self, experiment):
        self._values["step"] = experiment.trainer.current_steps
        wandb.log(self._values)
        self._values.clear()

    def on_val_batch_stop(self, experiment):
        self._values["val_step"] = self._validation_batches
        wandb.log(self._values)
        self._values.clear()
        self._validation_batches += 1
