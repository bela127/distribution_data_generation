import tensorflow as tf
from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource


class ChaoticDataSource(DataSource):
    def __init__(self, in_dim=3):
        if in_dim < 3:
            raise ValueError("in_dim of Chaotic data source must be at least 3")
        self.pool = ContinuousVectorPool(dim=in_dim,
                                         ranges=[[(-1, 1)]] * in_dim)
        self.point_shape = self.pool.get_shape()
        self.value_shape = self.point_shape

    @tf.function
    def _query(self, actual_query: tf.Tensor):
        a = 1.3
        one = 1
        b = 0.3

        out_x = [0.0, 0.0]

        x_ustacked = tf.unstack(actual_query)
        last_x = x_ustacked[1]
        last_last_x = x_ustacked[0]
        for i in x_ustacked[2:]:
            out_x.append(one - a * last_x + b * last_last_x)
            last_last_x = last_x
            last_x = i
        return actual_query, tf.stack(out_x)

    def possible_queries(self):
        return self.pool
