import random
from typing import Tuple

import tensorflow as tf
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class ZDataSource(DataSource):
    def __init__(self, dim: int, dependency_dimension: int = 1):
        self.dependency_dimension = dependency_dimension
        self.pool = ContinuousVectorPool(dim=dim * dependency_dimension, ranges=[[(0, 1)]] * dim * dependency_dimension)
        self.point_shape = (dim,)
        self.value_shape = (dim * dependency_dimension,)

    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        entries = tf.unstack(actual_queries)
        out = []

        next_value = None
        for entry in entries:
            for i in range(0, self.dependency_dimension):
                f1 = lambda : .0
                f2 = lambda : 1.
                f3 = lambda : entry

                rand = tf.random.uniform([])

                next_value = tf.case([(rand < .333, f1), ( rand < .666, f2)], f3)

                out.append(next_value)

        return actual_queries, tf.stack(out)

    def possible_queries(self):
        return self.pool
