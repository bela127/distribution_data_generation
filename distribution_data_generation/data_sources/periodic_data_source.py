import tensorflow as tf
from distribution_data_generation.data_source import DataSource


class PeriodicDataSource(DataSource):
    def __init__(self):
        super().__init__()

    @tf.function
    def __query(self, actual_queries: tf.Tensor):
        return actual_queries, tf.math.sin(actual_queries)
