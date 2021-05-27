"""Various utilities used throughout the library."""

import zmq


class AverageMeter:
    """Keep an average of values. The values should support addition and
    division with numbers."""
    def __init__(self):
        self.reset()

    def __iadd__(self, other):
        if isinstance(other, AverageMeter):
            self._value = other._value
            self._value_sum += other._value_sum
            self._count += other._count
        else:
            self._value = other
            self._value_sum += other
            self._count += 1
        return self

    def reset(self):
        self._value = None
        self._value_sum = 0.0
        self._count = 0

    @property
    def current_value(self):
        return self._value

    @property
    def average_value(self):
        return self._value_sum / self._count


def ddp_coordinate(uri, our_hostname):
    """Talk to the ddp configuration server to arange for the master host in
    order to be able to configure torch distributed.

    Arguments
    ---------
        uri: The uri to the configuration server
        our_hostname: The uri for others to connect to if we are the master
                      host

    Return
    ------
        - The uri to the master host
        - Our rank
        - The world size
    """
    # Connect to the ddp server
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.connect(uri)

    # Register our selves and ask for the master and our rank
    socket.send(bytes(our_hostname, "utf-8"))
    message = str(socket.recv(), "utf-8")

    # parse the message and return the information
    master, rank, world_size = message.split()

    return master, int(rank), int(world_size)


def rank_zero_only(f):
    """Decorator that executes f only if the currently active experiment has
    rank 0. This is used with distributed data parallel to train on multiple
    nodes."""
    def inner(*args, **kwargs):
        if Experiment.active().rank != 0:
            return

        return f(*args, **kwargs)
    return inner


def rank_zero_method(f):
    """Decorator that executes f only if the passed experiment has rank 0. This
    is used with distributed data parallel to train on multiple nodes."""
    def inner(experiment):
        if experiment.rank != 0:
            return
        return f(experiment)
    return inner
