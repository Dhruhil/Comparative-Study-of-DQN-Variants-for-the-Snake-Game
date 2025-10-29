# src/utils/schedules.py

import math

class LinearSchedule:
    """Linearly decay a parameter from start to end."""
    def __init__(self, start, end, duration):
        self.start = start
        self.end = end
        self.duration = duration

    def get_value(self, step):
        fraction = min(step / self.duration, 1.0)
        return self.start + fraction * (self.end - self.start)



class ExponentialSchedule:
    """Exponentially decay a parameter (e.g., epsilon) over time."""

    def __init__(self, start: float, end: float, decay_rate: float):
        self.start = start
        self.end = end
        self.decay_rate = decay_rate

    def get_value(self, step: int) -> float:
        value = self.end + (self.start - self.end) * math.exp(-1.0 * step / self.decay_rate)
        return max(self.end, value)
