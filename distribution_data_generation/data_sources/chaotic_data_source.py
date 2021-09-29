from distribution_data_generation.data_source import DataSource
import tensorflow as tf


class ChaoticDataSource(DataSource):
    def __init__(self):
        super().__init__()

    @tf.function
    def __query(self, actual_query: tf.Tensor):

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
