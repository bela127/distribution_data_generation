import random
from typing import Tuple

from distribution_data_generation.data_source import DataSource
import tensorflow as tf


class DoubleLinearDataSource(DataSource):
    def __init__(self, dependency_dimension: int = 1, factor: float = .5):
        self.dependency_dimension = dependency_dimension
        self.factor = factor

    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        entries = tf.unstack(actual_queries)
        out = []

        for entry in entries:
            for i in range(0, self.dependency_dimension):
                if bool(random.getrandbits(1)):
                    out.append(entry * self.factor)
                else:
                    out.append(entry)

        return actual_queries, tf.stack(out)
