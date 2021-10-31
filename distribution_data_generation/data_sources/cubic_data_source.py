from typing import Tuple

from distribution_data_generation.data_source import DataSource
import tensorflow as tf


class CubicDataSource(DataSource):
    def __init__(self, dependency_dimension: int = 1):
        self.dependency_dimension = dependency_dimension

    #TODO: range check -1, 1
    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        entries = tf.unstack(actual_queries)
        out = []

        for entry in entries:
            current_sum = 0.0
            for i in range(0, self.dependency_dimension):
                current_sum += entry
                out.append(tf.math.pow(current_sum, 3))

        return actual_queries, tf.stack(out)
