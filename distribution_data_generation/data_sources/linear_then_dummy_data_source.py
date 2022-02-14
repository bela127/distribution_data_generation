from typing import Tuple

import tensorflow as tf
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class LinearThenDummyDataSource(DataSource):
    def __init__(self, dim: int, dependency_dimension: int = 1):
        self.dependency_dimension = dependency_dimension
        self.pool = ContinuousVectorPool(dim=dim * dependency_dimension, ranges=[[(0, 1)]] * dim * dependency_dimension)
        self.point_shape = (dim,)
        self.value_shape = (dim * dependency_dimension,)

    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        entries = tf.unstack(actual_queries)
        out = []

        for entry in entries:
            out.append(entry)
            for i in range(1, self.dependency_dimension):
                out.append(0.5)

        return actual_queries, tf.stack(out)

    def possible_queries(self):
        return self.pool
