from typing import Callable

from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource
import tensorflow as tf


class GraphDataSource(DataSource):
    function = None

    def __init__(self, in_dim:int, graph: Callable = (lambda x: x)):
        self.function = tf.function(graph)
        self.pool = ContinuousVectorPool(dim=in_dim,
                                         ranges=[[(-1, 1)]] * in_dim)

    def _query(self, actual_queries: tf.Tensor):
        return actual_queries, self.function(actual_queries)

    def possible_queries(self):
        return self.pool
