import random
from typing import Tuple

from distribution_data_generation.data_source import DataSource
import tensorflow as tf


class InvZDataSource(DataSource):
    def __init__(self, dependency_dimension: int = 1):
        self.dependency_dimension = dependency_dimension

    # TODO: range check 0, 1
    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        entries = tf.unstack(actual_queries)
        out = []

        next_value = None
        for entry in entries:
            for i in range(0, self.dependency_dimension):
                # there are four cases, so we need to pick two random bits
                case = random.randint(0, 2)

                if case == 0:
                    next_value = .0
                elif case == 1:
                    next_value = 1.0
                else:
                    next_value = (entry - 0.5) * (-1) + 0.5

                out.append(next_value)

        return actual_queries, tf.stack(out)
