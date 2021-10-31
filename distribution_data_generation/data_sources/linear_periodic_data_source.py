from typing import Tuple

from distribution_data_generation.data_source import DataSource
import tensorflow as tf


class LinearPeriodicDataSource(DataSource):
    def __init__(self, period: float = .5):
        self.period = period

    @tf.function
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        entries = tf.unstack(actual_queries)
        out = []

        for entry in entries:
            out.append(entry % self.period)

        return actual_queries, tf.stack(out)
