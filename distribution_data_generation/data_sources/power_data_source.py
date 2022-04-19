import tensorflow as tf
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class PowerDataSource(DataSource):
    def __init__(self, dim: int, power: float = 2, dependency_dimension=1):
        self.power = power
        self.dim = dim
        self.dependency_dim = dependency_dimension
        self.pool = ContinuousVectorPool(dim=dim, ranges=[[(0, 1)]] * dim * dependency_dimension)
        self.point_shape = (dim,)
        self.value_shape = (dim * dependency_dimension,)

    @tf.function
    def _query(self, actual_query: tf.Tensor):
        res = tf.math.pow(actual_query, self.power)
        res = tf.repeat(res, repeats=[self.dependency_dim] * self.dim, axis=0)
        return actual_query, res

    def possible_queries(self):
        return self.pool
