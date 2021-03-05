import os
from os import path

import torch

from .base import Callback


class ModelCheckpoint(Callback):
    """ModelCheckpoint is responsible for saving the model and the optimizer
    as well as resuming from the latest """
    def __init__(
        self,
        model_path:str = "{:06d}.ckpt",
        resume_from_checkpoint:bool = True,
        save_optimizer:bool = True,
        save_frequency:int = 1
    ):
        self.model_path = model_path
        self.resume_from_checkpoint = resume_from_checkpoint
        self.save_optimizer = save_optimizer
        self.save_frequency = save_frequency

    def on_train_start(self, experiment):
        if not self.resume_from_checkpoint:
            return

        checkpoint_dir = path.join(
            experiment.arguments["output_dir"],
            "checkpoints"
        )
        if not path.exists(checkpoint_dir):
            return

        checkpoints = os.listdir(checkpoint_dir)
        if len(checkpoints) == 0:
            return

        last_checkpoint = path.join(checkpoint_dir, sorted(checkpoints)[-1])
        if experiment.arguments["verbose"] > 0:
            print("Loading from checkpoint: {!r}".format(last_checkpoint))
        data = torch.load(last_checkpoint, map_location="cpu")

        experiment.trainer.set_epoch(experiment, data["epoch"])
        experiment.trainer.set_steps(experiment, data["steps"])
        experiment.model.load_state_dict(data["model_state"])
        if "optimizer_state" in data:
            if experiment.arguments["verbose"] > 0:
                print("Loading optimizer from checkpoint")
            experiment.optimizer.load_state_dict(data["optimizer_state"])

    def on_epoch_stop(self, experiment):
        if (experiment.trainer.current_epoch % self.save_frequency) != 0:
            return

        data = {}
        data["epoch"] = experiment.trainer.current_epoch
        data["steps"] = experiment.trainer.current_steps
        data["model_state"] = experiment.model.state_dict()
        if self.save_optimizer:
            data["optimizer_state"] = experiment.optimizer.state_dict()

        checkpoint_file = path.join(
            experiment.arguments["output_dir"],
            "checkpoints",
            self.model_path.format(data["steps"])
        )
        if not path.exists(path.dirname(checkpoint_file)):
            os.makedirs(path.dirname(checkpoint_file))
        torch.save(data, checkpoint_file)
