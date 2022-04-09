import math
from typing import Tuple

import tensorflow as tf
from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class LinearStepDataSource(DataSource):
    def __init__(self, dim: int, step_size: float = .5):
        self.step_size = math.fabs(step_size)
        self.half_step = step_size / 2
        self.pool = ContinuousVectorPool(dim=dim, ranges=[[(0, 1)]] * dim)
        self.point_shape = (dim,)
        self.value_shape = (dim,)


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

    def possible_queries(self):
        return self.pool
