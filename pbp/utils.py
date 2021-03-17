"""Various utilities used throughout the library."""


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
