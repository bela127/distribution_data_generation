from typing import Tuple

import tensorflow as tf
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class CubicDataSource(DataSource):
    def __init__(self, in_dim: int, dependency_dimension: int = 1):
        self.dependency_dimension = dependency_dimension
        self.pool = ContinuousVectorPool(dim=in_dim * dependency_dimension,
                                         ranges=[[(-1, 1)]] * in_dim * dependency_dimension)
        self.point_shape = (in_dim,)
        self.value_shape = (in_dim * dependency_dimension,)

    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        entries = tf.unstack(actual_queries)
        out = []

        for entry in entries:
            current_sum = 0.0
            for i in range(0, self.dependency_dimension):
                current_sum += entry
                out.append(tf.math.pow(current_sum, 3))

        return actual_queries, tf.stack(out)

    def possible_queries(self):
        return self.pool
