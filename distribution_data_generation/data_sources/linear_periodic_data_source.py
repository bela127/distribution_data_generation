from typing import Tuple

import tensorflow as tf
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class LinearPeriodicDataSource(DataSource):
    def __init__(self, dim: int, period: float = .5):
        self.period = period
        self.pool = ContinuousVectorPool(dim=dim, ranges=[[(0, 1)]] * dim)
        self.point_shape = (dim,)
        self.value_shape = (dim,)

    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        entries = tf.unstack(actual_queries)
        out = []

        for entry in entries:
            out.append(entry % self.period)

        return actual_queries, tf.stack(out)

    def possible_queries(self):
        return self.pool
