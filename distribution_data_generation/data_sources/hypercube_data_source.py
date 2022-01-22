import random
from typing import Tuple

import tensorflow as tf
from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


# TODO: this is not possible
class HypercubeDataSource(DataSource):
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
            next_entry = None

            non_random_entry = random.randint(0, self.dependency_dimension - 1)
            for i in range(0, self.dependency_dimension):
                if i == non_random_entry:
                    next_entry = random.uniform(.0, 1.0)
                else:
                    next_entry = .0 if bool(random.getrandbits(1)) else 1.0
                out.append(next_entry)

        return actual_queries, tf.stack(out)

    def possible_queries(self):
        return self.pool
