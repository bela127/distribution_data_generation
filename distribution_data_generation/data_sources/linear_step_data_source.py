import math
from typing import Tuple

from distribution_data_generation.data_source import DataSource
import tensorflow as tf


class LinearStepDataSource(DataSource):
    def __init__(self, step_size: float = .5):
        self.step_size = math.fabs(step_size)
        self.half_step = step_size / 2

    # TODO: range check 0-1
    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        entries = tf.unstack(actual_queries)
        out = []

        next_entry = None
        for entry in entries:
            dif = (entry % self.step_size)
            if (dif - self.half_step) < 0:
                next_entry = entry - dif
            else:
                next_entry = entry + (self.step_size - dif)
            out.append(next_entry)

        return actual_queries, tf.stack(out)
