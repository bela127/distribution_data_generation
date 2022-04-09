import random
from typing import Tuple

from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool
import tensorflow as tf

from distribution_data_generation.data_source import DataSource


class PlusDataSource(DataSource):
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
            random_dim = tf.random.uniform(shape=(), minval=0, maxval=self.dependency_dimension, dtype=tf.int32)
            for i in range(0, self.dependency_dimension):

                if entry == 0.5 and i == random_dim:
                    next_value = tf.random.uniform([])
                else:
                    next_value = 0.5

                out.append(next_value)

        return actual_queries, tf.stack(out)

    def possible_queries(self):
        return self.pool
