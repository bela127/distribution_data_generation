from distribution_data_generation.data_source import DataSource

import tensorflow as tf


class PowerDataSource(DataSource):
    def __init__(self, power: float = 2):
        self.power = power

    @tf.function
    def _query(self, actual_query: tf.Tensor):
        return actual_query, tf.math.pow(actual_query, self.power)
