import random
from typing import Tuple

import tensorflow as tf
from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class CrossDataSource(DataSource):

    def __init__(self, in_dim: int, dependency_dimension: int = 1):
        self.dependency_dimension = dependency_dimension
        self.pool = ContinuousVectorPool(dim=in_dim * dependency_dimension,
                                         ranges=[[(0, 1)]] * in_dim * dependency_dimension)
        self.point_shape = (in_dim,)
        self.value_shape = (in_dim,)

    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        entries = tf.unstack(actual_queries)
        out = []
        e = None
        for entry in entries:
            for i in range(0, self.dependency_dimension):
                f1 = lambda: 1 - entry
                f2 = lambda: entry
                e = tf.case([(tf.random.uniform([]) < 0.5, f1)], f2)
                out.append(e)
        return actual_queries, tf.stack(out)

    def possible_queries(self):
        return self.pool
