import tensorflow as tf
from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class PowerDataSource(DataSource):
    def __init__(self, dim: int, power: float = 2):
        self.power = power
        self.pool = ContinuousVectorPool(dim=dim, ranges=[[(0, 1)]] * dim)
        self.point_shape = (dim,)
        self.value_shape = (dim,)

    @tf.function
    def _query(self, actual_query: tf.Tensor):
        return actual_query, tf.math.pow(actual_query, self.power)

    def possible_queries(self):
        return self.pool
