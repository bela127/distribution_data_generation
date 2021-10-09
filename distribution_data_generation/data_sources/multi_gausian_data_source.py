import random
from typing import Tuple

from distribution_data_generation.data_source import DataSource
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
import tensorflow as tf


class MultiGausianDataSource(DataSource):
    def __init__(self, in_dim: int, out_dim: int, min_x: float = -10.0, max_x: float = 10.0):
        super().__init__()
        x = tf.random.uniform((in_dim,), minval=min_x, maxval=max_x)
        y = tf.random.uniform((out_dim,), minval=-3.0,
                              maxval=3.0)  # with all default values, 3 is about the highest sample you can get
        self.gpr = GaussianProcessRegressor(kernel=RationalQuadratic())
        self.gpr.fit([x], [y])

    # @tf.function
    # there is an issue with using numpy methods in tf graphs. gpr uses numpy methods. looking into solutions for this
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        y = self.gpr.sample_y([actual_queries], random_state=random.randint(10, 10000000))
        y = tf.constant([a[0] for a in y[0]])  # strange dimension issues
        self.gpr.fit([actual_queries], [y])
        return actual_queries, y
