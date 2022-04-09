import tensorflow as tf
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class RandomDataSource(DataSource):
    def __init__(self, dim: int):
        self.pool = ContinuousVectorPool(dim=dim, ranges=[[(0, 1)]] * dim)
        self.point_shape = (dim,)
        self.value_shape = (dim,)

    @tf.function
    def _query(self, x: tf.Tensor):
        return x, tf.random.uniform(x.shape)

    def possible_queries(self):
        return self.pool
