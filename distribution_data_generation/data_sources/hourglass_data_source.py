import random
from typing import Tuple

import tensorflow as tf
from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class HourglassDataSource(DataSource):
    def __init__(self, in_dim: int, dependency_dimension: int = 1):
        self.dependency_dimension = dependency_dimension
        self.pool = ContinuousVectorPool(dim=in_dim * dependency_dimension,
                                         ranges=[[(0, 1)]] * in_dim * dependency_dimension)

        self.point_shape = (in_dim,)
        self.value_shape = (in_dim * dependency_dimension,)

    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        entries = tf.unstack(actual_queries)
        out = []

        for entry in entries:
            for i in range(0, self.dependency_dimension):
                next_value = None
                f1 = lambda : (entry - 0.5) * (-1) + 0.5
                f2 = lambda : .0
                f3 = lambda : 1.
                f4 = lambda : entry
                rand = tf.random.uniform([])
                next_value = tf.case([(rand < .25, f1), (rand < .5, f2), (rand < .75, f3)], f4)

                out.append(next_value)

        return actual_queries, tf.stack(out)

    def possible_queries(self):
        return self.pool
