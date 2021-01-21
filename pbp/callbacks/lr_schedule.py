from .base import Callback


class SingleGroupLRScheduler(Callback):
    def on_train_start(self, experiment):
        if len(experiment.optimizer.param_groups) > 1:
            raise RuntimeError(("{} only supports a single parameter group "
                                "for a single optimizer.").format(type(self)))
        param_group = experiment.optimizer.param_groups[0]
        if "initial_lr" not in param_group:
            param_group["initial_lr"] = param_group["lr"]

    def _report_lr_change(self, param_group):
        if self.verbose:
            print("Setting LR to {}".format(param_group["lr"]))


class WarmupLR(SingleGroupLRScheduler):
    def __init__(
        self,
        lr_warmup_steps:int = 0,
        lr_schedule_verbose:bool = False
    ):
        self.warmup_steps = lr_warmup_steps
        self.verbose = lr_schedule_verbose

    def on_train_batch_start(self, experiment):
        current_steps = experiment.trainer.current_steps
        if current_steps < self.warmup_steps:
            percentage = current_steps / (self.warmup_steps - 1 + 1e-6)
            param_group = experiment.optimizer.param_groups[0]
            param_group["lr"] = param_group["initial_lr"] * percentage

            self._report_lr_change(param_group)


class MultiplicativeLRSchedule(SingleGroupLRScheduler):
    def __init__(
        self,
        lr_mul_schedule:str = "",
        lr_schedule_verbose:bool = False
    ):
        self.verbose = lr_schedule_verbose

        # Parse the learning rate schedule from the argument
        try:
            steps, factors = zip(*[
                p.split("-") for p in lr_mul_schedule.split(",")
            ])
            factors = list(map(float, factors))
            steps = list(map(int, steps))
        except ValueError:
            steps, factors = [], []

        self.steps = steps
        self.factors = factors
        self._step_idx = 0

    def on_train_batch_start(self, experiment):
        if self._step_idx >= len(self.steps):
            return

        current_steps = experiment.trainer.current_steps
        if current_steps < self.steps[self._step_idx]:
            return

        # Change the learning rate according to the multiplicative factor
        param_group = experiment.optimizer.param_groups[0]
        param_group["lr"] = param_group["initial_lr"] * self.factors[self._step_idx]
        self._step_idx += 1

        self._report_lr_change(param_group)
