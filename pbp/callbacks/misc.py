import json
import os
from os import path
from subprocess import run

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
