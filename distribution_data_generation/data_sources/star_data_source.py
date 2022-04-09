import random
from typing import Tuple

import tensorflow as tf
from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource
from distribution_data_generation.data_sources.cross_data_source import CrossDataSource
from distribution_data_generation.data_sources.plus_data_source import PlusDataSource


class StarDataSource(DataSource):
    def __init__(self, dim: int, dependency_dimension: int = 1):
        self.dependency_dimension = dependency_dimension
        self.pool = ContinuousVectorPool(dim=dim * dependency_dimension, ranges=[[(0, 1)]] * dim * dependency_dimension)
        self.point_shape = (dim,)
        self.value_shape = (dim * dependency_dimension,)
        self.plus = PlusDataSource(1, dependency_dimension)
        self.ex = CrossDataSource(1,dependency_dimension)

    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        entries = tf.unstack(actual_queries)
        out = tf.convert_to_tensor([])
        p = (1 + self.dependency_dimension) / (1 + 3 * self.dependency_dimension)


        next_value = None
        for entry in entries:
            f1 = lambda : self.ex._query([entry])
            f2 = lambda : self.plus._query([entry])

            a, next_value = tf.case([(tf.random.uniform([]) < p, f2)], f1)

            out = tf.concat([out, next_value], 0)

        return actual_queries, tf.stack(out)

    def possible_queries(self):
        return self.pool
