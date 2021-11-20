from typing import List

from active_learning_ts.pools.discrete_vector_pool import DiscreteVectorPool
from active_learning_ts.pools.retrievement_strategy import RetrievementStrategy

from distribution_data_generation.data_source import DataSource
import tensorflow as tf


class DataSetDataSource(DataSource):
    """
    This solution sucks. tf.dataset isn't meant for this, neither is tf..hashtable. This can be changed in the future,
    i think there are tf based k-d-tree implementations one could use here
    """
    data_points = None
    data_values = None

    def __init__(self, in_dim: int, data_points: List[tf.Tensor], data_values: List[tf.Tensor]):
        self.data_points = data_points
        self.data_values = data_values
        self.in_dim = in_dim
        self.pool = None

    def post_init(self, retrievement_strategy: RetrievementStrategy):
        super().post_init(retrievement_strategy)
        self.pool = DiscreteVectorPool(in_dim=self.in_dim, queries=self.data_points,
                                       find_streategy=self.retrievementStrategy)


    def _query(self, actual_queries: tf.Tensor):
        # TODO: implement this with a kd-tree
        place = 0
        for t in self.data_points:
            if tf.reduce_all(tf.math.equal(t, actual_queries)):
                break
            place += 1
        return actual_queries, self.data_values[place]

    def possible_queries(self):
        return self.pool
