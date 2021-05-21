import json
import os
from os import path
from subprocess import run
import sys

from .base import Callback


class LogArguments(Callback):
    """LogArguments logs the experiment arguments and git hash if available in
    order to have reproducible runs."""
    def on_train_start(self, experiment):
        arguments = dict(experiment.arguments.items())
        arguments["git_hash"] = self._get_git_hash()
        arguments["git_dirty"] = self._get_git_dirty()
        del arguments["experiment"]

        with open(path.join(arguments["output_dir"], "args.json"), "w") as f:
            json.dump(arguments, f, indent=4)

    def _get_git_hash(self):
        dirname = path.dirname(self._get_top_level_file())
        try:
            r = run(
                ["git", "rev-parse", "HEAD"],
                cwd=dirname,
                capture_output=True
            )
            return r.stdout.decode("utf-8").strip()
        except FileNotFoundError:
            return None

    def _get_git_dirty(self):
        dirname = path.dirname(self._get_top_level_file())
        try:
            r = run(
                ["git", "diff", "--quiet"],
                cwd=dirname
            )
            return r.returncode == 1
        except FileNotFoundError:
            return None

    def _get_top_level_file(self):
        import __main__

        return path.join(
            os.getcwd(),
            __main__.__file__
        )


class CooperativeGridScheduling(Callback):
    """Exit with a non-zero exit code after a given amount of epochs or
    iterations in order to allow the scheduler to rebalance.

    NOTE: This callback does not resubmit the job to the scheduler. This should
          be done externally.

    Arguments
    ---------
        yield_after_iterations: int, exit after that many training iterations
                                (default: 0)
        yield_after_epochs: int, exit after that many epochs (default: 0)
        exit_code: int, which exit code to use when exiting (default: 1)
    """
    def __init__(self, yield_after_iterations:int = 0,
                 yield_after_epochs:int = 0, exit_code:int = 1):
        self.yield_after_iterations = yield_after_iterations
        self.yield_after_epochs = yield_after_epochs
        self.exit_code = exit_code

        self._iterations = 0
        self._epochs = 0

    def _yield(self):
        sys.exit(self.exit_code)

    def on_train_batch_stop(self, experiment):
        if self.yield_after_iterations <= 0:
            return

        self._iterations += 1
        if self._iterations >= self.yield_after_iterations:
            self._yield()

    def on_epoch_start(self, experiment):
        if self.yield_after_epochs <= 0:
            return

        self._epochs += 1
        if self._epochs > self.yield_after_epochs:
            self._yield()
