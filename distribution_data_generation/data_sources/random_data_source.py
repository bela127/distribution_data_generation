from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource

import tensorflow as tf


class RandomDataSource(DataSource):
    def __init__(self, dim: int):
        self.pool = ContinuousVectorPool(dim=dim, ranges=[[(0, 1)]] * dim)

    @tf.function
    def _query(self, x: tf.Tensor):
        return x, tf.random.uniform(x.shape)

    def possible_queries(self):
        return self.pool
