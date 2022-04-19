from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool
import tensorflow as tf
from distribution_data_generation.data_source import DataSource


class LinearDataSource(DataSource):
    def __init__(self, dim: int, dependency_dim: int = 1):
        self.dim = dim
        self.dependency_dim = dependency_dim
        self.pool = ContinuousVectorPool(dim=dim, ranges=[[(0, 1)]] * dim * dependency_dim)
        self.point_shape = (dim,)
        self.value_shape = (dim * dependency_dim,)

    def _query(self, actual_queries):
        result = actual_queries
        result = tf.repeat(result, repeats=[self.dependency_dim] * self.dim, axis=0)
        return actual_queries, result

    def possible_queries(self):
        return self.pool
