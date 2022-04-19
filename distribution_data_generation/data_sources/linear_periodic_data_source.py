from typing import Tuple

import tensorflow as tf
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class LinearPeriodicDataSource(DataSource):
    def __init__(self, dim: int, period: float = .5, dependency_dim: int = 1):
        self.period = period
        self.dim = dim
        self.dependency_dim = dependency_dim
        self.pool = ContinuousVectorPool(dim=dim, ranges=[[(0, 1)]] * dim * dependency_dim)
        self.point_shape = (dim,)
        self.value_shape = (dim* dependency_dim,)

    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        entries = tf.unstack(actual_queries)
        out = []
        for entry in entries:
            out.append(entry % self.period)

        result = tf.stack(out)
        result = tf.repeat(result, repeats=[self.dependency_dim] * self.dim, axis=0)
        return actual_queries, result

    def possible_queries(self):
        return self.pool
