import math

from distribution_data_generation.data_source import DataSource

import tensorflow as tf


class SineDataSource(DataSource):
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    @tf.function
    def _query(self, actual_query: tf.Tensor):
        return actual_query, (tf.math.sin(actual_query * 2 * math.pi * self.scale) + 1) / 2
