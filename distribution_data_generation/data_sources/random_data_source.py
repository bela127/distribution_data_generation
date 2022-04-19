import tensorflow as tf
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class RandomDataSource(DataSource):
    def __init__(self, dim: int, out_dim: int = None):
        out_dim = dim if out_dim is None else out_dim
        self.pool = ContinuousVectorPool(dim=dim, ranges=[[(0, 1)]] * dim)
        self.point_shape = (dim,)
        self.value_shape = (out_dim,)

    @tf.function
    def _query(self, x: tf.Tensor):
        return x, tf.random.uniform(self.value_shape)

    def possible_queries(self):
        return self.pool
