import random
from typing import Tuple

from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource
import tensorflow as tf


class NonCoexistenceDataSource(DataSource):
    def __init__(self, dim: int):
        self.pool = ContinuousVectorPool(dim=dim, ranges=[[(0, 1)]] * dim)

    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        entries = tf.unstack(actual_queries)
        out = []

        non_zero_index = random.randint(0, len(entries))
        i = 0
        for entry in entries:
            if i == non_zero_index:
                out.append(entry)
            else:
                out.append(0.0)
            i += 1

        return actual_queries, tf.stack(out)

    def possible_queries(self):
        return self.pool
