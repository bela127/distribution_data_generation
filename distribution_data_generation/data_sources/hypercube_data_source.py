import random
from typing import Tuple

import tensorflow as tf
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class HypercubeDataSource(DataSource):
    def __init__(self, dim: int):
        self.pool = ContinuousVectorPool(dim=dim, ranges=[[(0, 1)]] * dim)

        self.point_shape = (dim,)
        self.value_shape = (dim,)

    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

        entries = tf.unstack(actual_queries)
        out = []

        for entry in entries:
            next_entry = None

            rand = tf.random.uniform([])
            f1 = lambda : rand
            f2 = lambda : .0
            f3 = lambda : 1.

            next_entry = tf.case([(entry == 1., f1), (entry == .0, f1), (rand < .5, f2)], f3)

            out.append(next_entry)

        return actual_queries, tf.stack(out)


def possible_queries(self):
    return self.pool
