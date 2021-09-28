from distribution_data_generation.data_source import DataSource

import tensorflow as tf


class RandomDataSource(DataSource):
    def __init__(self):
        super().__init__()

    @tf.function
    def query(self, x: tf.Tensor):
        return tf.random.uniform(x.shape)