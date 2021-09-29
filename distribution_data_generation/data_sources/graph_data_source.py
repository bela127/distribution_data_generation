from typing import Callable

from distribution_data_generation.data_source import DataSource
import tensorflow as tf


class GraphDataSource(DataSource):
    function = None

    def __init__(self, graph: Callable = (lambda x: x)):
        super().__init__()
        self.function = tf.function(graph)

    def _query(self, actual_queries: tf.Tensor):
        return actual_queries, self.function(actual_queries)
