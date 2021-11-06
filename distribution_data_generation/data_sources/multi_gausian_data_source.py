import random
from typing import Tuple

from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool

from distribution_data_generation.data_source import DataSource
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
import tensorflow as tf


class MultiGausianDataSource(DataSource):
    def __init__(self, in_dim: int, out_dim: int, min_x: float = -10.0, max_x: float = 10.0):
        x = tf.random.uniform((in_dim,), minval=min_x, maxval=max_x)
        y = tf.random.uniform((out_dim,), minval=-3.0,
                              maxval=3.0)  # with all default values, 3 is about the highest sample you can get
        self.gpr = GaussianProcessRegressor(kernel=RationalQuadratic())
        self.gpr.fit([x], [y])
        self.queries = []
        self.values = []
        self.pool = ContinuousVectorPool(dim=in_dim, ranges=[[(min_x, max_x)]])

    # @tf.function
    # there is an issue with using numpy methods in tf graphs. gpr uses numpy methods. looking into solutions for this
    def _query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # TODO: in order to learn properly here, every point has to be trained every time. Overriding query directly \
        #  would avoid this
        y = self.gpr.sample_y([actual_queries], random_state=random.randint(10, 10000000))
        y = tf.constant([a[0] for a in y[0]])  # strange dimension issues
        self.queries.append(actual_queries)
        self.values.append(y)
        self.gpr.fit(self.queries, self.values)

        return actual_queries, y

    def possible_queries(self):
        return self.pool

