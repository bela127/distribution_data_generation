import tensorflow as tf
from active_learning_ts.data_retrievement.pools.discrete_vector_pool import DiscreteVectorPool
from active_learning_ts.data_retrievement.retrievement_strategy import RetrievementStrategy

from distribution_data_generation.data_source import DataSource


class DataSetDataSource(DataSource):
    data_points = None
    data_values = None

    def __init__(self, data_points: tf.Tensor, data_values: tf.Tensor):
        """
        Creates basically a tensor dictionary.
        This version works with any dimension, but only the storing of 1-D tensors is allowed

        :param data_points: A 2-D Tensor. Acts as a list of vectors
        :param data_values: A 2-D Tensor. Acts as a list of vectors. The first dimension of data_values must be the
                same as that of data_points
        """
        self.data_points = data_points
        self.data_values = data_values

        if not len(data_points.shape) == 2:
            raise ValueError("Requires vector input")

        self.point_shape = data_points[0].shape
        self.value_shape = data_values[0].shape
        self.pool = None

    def post_init(self, retrievement_strategy: RetrievementStrategy):
        super().post_init(retrievement_strategy)
        self.pool = DiscreteVectorPool(in_dim=self.point_shape[0], queries=self.data_points,
                                       retrievement_strategy=self.retrievement_strategy)

    def query(self, actual_queries: tf.Tensor):
        return tf.gather(self.data_points, actual_queries), tf.gather(self.data_values, actual_queries)

    def possible_queries(self):
        return self.pool
