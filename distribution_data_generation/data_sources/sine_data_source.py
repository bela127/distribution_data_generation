import math

import tensorflow as tf
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class SineDataSource(DataSource):
    def __init__(self, dim: int, dependency_dim=1, scale: float = 1.0):
        self.scale = scale
        self.dependency_dim = dependency_dim
        self.dim = dim
        self.pool = ContinuousVectorPool(dim=dim, ranges=[[(0, 1)]] * dim * dependency_dim)
        self.point_shape = (dim,)
        self.value_shape = (dim * dependency_dim,)

    @tf.function
    def _query(self, actual_query: tf.Tensor):
        res = (tf.math.sin(actual_query * 2 * math.pi * self.scale) + 1) / 2
        res = tf.repeat(res, repeats=[self.dependency_dim] * self.dim, axis=0)
        return actual_query, res

    def possible_queries(self):
        return self.pool
