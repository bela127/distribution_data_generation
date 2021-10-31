import random
from typing import Tuple

from distribution_data_generation.data_source import DataSource
import tensorflow as tf


class LinearThenNoiseDataSource(DataSource):
    def __init__(self, dependency_dimension: int = 1):
        self.dependency_dimension = dependency_dimension

    # TODO: range check 0, 1
    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        entries = tf.unstack(actual_queries)
        out = []

        next_entry = None
        for entry in entries:
            out.append(entry)
            for i in range(1, self.dependency_dimension):
                next_entry = random.uniform(0, 1)
                out.append(next_entry)

        return actual_queries, tf.stack(out)
