"""Launch a configuration server that can distribute rank and master
information to its clients in order to run pytorch's DDP."""

import argparse
from collections import defaultdict

import zmq


class DistributedRun:
    def __init__(self):
        self._id = 0
        self._master = None

    def add_client(self, host):
        if self._master is None:
            self._master = host

        rank = self._id
        self._id += 1

        return self._master, rank


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=("Launch a configuration server that can distribute rank "
                     "and master information to its clients in order to run "
                     "pytorch's DDP.")
    )
    parser.add_argument(
        "world_size",
        type=int,
        help="Define the number of nodes in our distributed setup"
    )
    parser.add_argument(
        "--bind_address",
        default="tcp://*:1234",
        help="Set the uri for the server to bind to (default: tcp://*:1234)"
    )
    parser.add_argument(
        "--persistent",
        action="store_true",
        help=("Wait indefinitely to distribute rank and master information "
              "for many runs")
    )

    args = parser.parse_args(argv)

    world_size = args.world_size
    assert world_size > 1

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(args.bind_address)

    try:
        rank = -1
        while args.persistent or rank < 0:
            rank = -1
            run = DistributedRun()
            while rank+1 < world_size:
                host = str(socket.recv(), "utf-8")
                master, rank = run.add_client(host)
                socket.send(bytes("{} {} {}".format(master, rank, world_size), "utf-8"))

    except KeyboardInterrupt:
        pass
