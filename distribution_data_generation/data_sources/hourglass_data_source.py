import random
from typing import Tuple

from distribution_data_generation.data_source import DataSource
import tensorflow as tf


class HourglassDataSource(DataSource):
    def __init__(self, dependency_dimension: int = 1):
        self.dependency_dimension = dependency_dimension

    # TODO: range check 0, 1
    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        entries = tf.unstack(actual_queries)
        out = []

        for entry in entries:
            for i in range(0, self.dependency_dimension):
                # there are four cases, so we need to pick two random bits
                next_value = None
                if bool(random.getrandbits(1)):
                    if bool(random.getrandbits(1)):
                        next_value = (entry - 0.5) * (-1) + 0.5
                    else:
                        next_value = 0.0
                else:
                    if bool(random.getrandbits(1)):
                        next_value = 1.0
                    else:
                        next_value = entry
                out.append(next_value)

        return actual_queries, tf.stack(out)
