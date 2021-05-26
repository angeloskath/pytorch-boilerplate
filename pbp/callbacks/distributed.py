import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ..utils import ddp_coordinate
from .base import Callback


class DistributedSetup(Callback):
    """Connect to the configuration server to receive the master host, our
    rank and the world size.

    Arguments
    ---------
        distributed_config_server: uri to the configuration server, setting it
                                   to empty disables distributed training
                                   (default: tcp://localhost:1234)
        distributed_our_hostname: uri for others to connect to us if we are the
                                  master host (default: tcp://localhost:5555)
        distributed_backend: str, defines the backend used for torch's
                             distributed process group (default: gloo)
    """
    def __init__(
        self,
        distributed_config_server:str = "tcp://localhost:1234",
        distributed_our_hostname:str = "tcp://localhost:5555",
        distributed_backend:str = "gloo"
    ):
        self.config_server = distributed_config_server
        self.our_hostname = distributed_our_hostname
        self.backend = distributed_backend

    def on_train_start(self, experiment):
        if self.config_server == "":
            return

        master, rank, world_size = ddp_coordinate(
            self.config_server,
            self.our_hostname
        )

        dist.init_process_group(
            backend=self.backend,
            init_method=master,
            rank=rank,
            world_size=world_size
        )

        experiment.rank = rank
        experiment.model = DistributedDataParallel(experiment.model)

    def on_train_stop(self, experiment):
        if self.config_server == "":
            return

        dist.destroy_process_group()
