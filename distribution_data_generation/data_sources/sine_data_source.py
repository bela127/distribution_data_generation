import math

import tensorflow as tf
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class SineDataSource(DataSource):
    def __init__(self, dim: int, scale: float = 1.0):
        self.scale = scale
        self.pool = ContinuousVectorPool(dim=dim, ranges=[[(0, 1)]] * dim)
        self.point_shape = (dim,)
        self.value_shape = (dim,)

    @tf.function
    def _query(self, actual_query: tf.Tensor):
        return actual_query, (tf.math.sin(actual_query * 2 * math.pi * self.scale) + 1) / 2

    def possible_queries(self):
        return self.pool
