import random
from typing import Tuple

import tensorflow as tf
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class StarDataSource(DataSource):
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
            non_random_dim = random.randint(0, self.dependency_dimension)
            for i in range(0, self.dependency_dimension):
                # there are four cases, so we need to pick two random bits

                if bool(random.getrandbits(1)):
                    next_value = (entry - 0.5) * (-1) + 0.5
                else:
                    next_value = entry

                if i == non_random_dim:
                    next_value = 0.5

                out.append(next_value)

        return actual_queries, tf.stack(out)

    def possible_queries(self):
        return self.pool
