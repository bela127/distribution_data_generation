import random
from typing import Tuple

import tensorflow as tf
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class HypercubeDataSource(DataSource):
    def __init__(self, dim: int, dependency_dim: int):
        self.pool = ContinuousVectorPool(dim=dim, ranges=[[(0, 1)]] * (dim * dependency_dim))
        self.dependency_dim = dependency_dim
        self.point_shape = (dim,)
        self.value_shape = (dim,)

    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        entries = tf.unstack(actual_queries)
        out = []

        for entry in entries:
            next_entry = None

            special_dim = tf.random.uniform(shape=(), minval=0, maxval=self.dependency_dim, dtype=tf.int32)

            for i in range(0, self.dependency_dim):
                if (entry == .0 or entry == 1.0) and i == special_dim:
                    next_entry = tf.random.uniform([])
                else:
                    next_entry = tf.case([(tf.random.uniform([]) < 0.5, lambda: 1.0)], lambda: .0)
                out.append(next_entry)

        return actual_queries, tf.stack(out)


def possible_queries(self):
    return self.pool
