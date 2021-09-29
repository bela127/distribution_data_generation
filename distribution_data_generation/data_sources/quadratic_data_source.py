from distribution_data_generation.data_source import DataSource

import tensorflow as tf


class QuadraticDataSource(DataSource):
    def __init__(self):
        super().__init__()

    @tf.function
    def _query(self, actual_query: tf.Tensor):
        return actual_query, actual_query * actual_query
