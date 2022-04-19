import math
from typing import Tuple

import tensorflow as tf
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class LinearStepDataSource(DataSource):
    def __init__(self, dim: int, step_size: float = .5, dependency_dim: int = 1):
        self.step_size = math.fabs(step_size)
        self.dim = dim
        self.dependency_dim = dependency_dim
        self.half_step = step_size / 2
        self.pool = ContinuousVectorPool(dim=dim, ranges=[[(0, 1)]] * dim * dependency_dim)
        self.point_shape = (dim,)
        self.value_shape = (dim * dependency_dim,)

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

        res = tf.stack(out)
        res = tf.repeat(res, repeats=[self.dependency_dim] * self.dim, axis=0)
        return actual_queries, res

    def possible_queries(self):
        return self.pool
